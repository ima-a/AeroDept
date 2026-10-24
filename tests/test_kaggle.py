from pathlib import Path
import ast
import importlib.util
import json
import zipfile

import h5py
import numpy as np
from PIL import Image
import pytest

from aerodepth_lite.kaggle import (discover_dataset_root, validate_dataset, grouped_manifest,
    load_or_create_manifest, validate_manifest, path_group)


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def weather_dataset(tmp_path):
    root = tmp_path / 'input' / 'midair-upload' / 'extra' / 'MidAir'
    for weather_index, weather in enumerate(('sunny', 'foggy')):
        parent = root / 'Kite_training' / weather
        parent.mkdir(parents=True)
        with h5py.File(parent / 'sensor_records.hdf5', 'w') as handle:
            for number in (3, 21, 4, 5, 6):
                trajectory = f'trajectory_{number + weather_index * 2000:04d}'
                camera_paths = {}
                for modality in ('color_left', 'depth'):
                    directory = parent / modality / trajectory
                    directory.mkdir(parents=True)
                    camera_paths[modality] = []
                    for frame in range(2):
                        path = directory / f'{frame:06d}.PNG'
                        array = np.zeros((12, 20, 3), dtype=np.uint8) if modality == 'color_left' else np.full((12, 20), 5, dtype=np.float16).view(np.uint16)
                        Image.fromarray(array).save(path)
                        camera_paths[modality].append(path.relative_to(root).as_posix())
                group = handle.create_group(trajectory)
                for modality, paths in camera_paths.items():
                    group.create_dataset(f'camera_data/{modality}', data=paths, dtype=h5py.string_dtype())
                for field in ('groundtruth/velocity', 'imu/accelerometer', 'imu/gyroscope'):
                    sensor = group.create_dataset(field, data=np.full((8, 3), weather_index + 1.0))
                    sensor.attrs['sampling_frequency'] = 100.0
    return root


def test_nested_root_and_complete_inventory(weather_dataset):
    root = discover_dataset_root(weather_dataset.parents[2])
    assert root == weather_dataset
    records, report = validate_dataset(root, expected_count=10)
    assert len(records) == 20
    assert report['expected_inventory_verified']
    assert len(report['hdf5_validation']['trajectories']) == 10
    with pytest.raises(ValueError, match='Expected'):
        validate_dataset(root, expected_count=11)
    with pytest.raises(ValueError, match='expected inventory'):
        validate_dataset(root, expected_trajectories=['missing'])


def test_grouped_splits_deterministic_and_pilot_exclusion(weather_dataset, tmp_path):
    records, _ = validate_dataset(weather_dataset)
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    payload = grouped_manifest(records, a)
    grouped_manifest(list(reversed(records)), b)
    assert a.read_bytes() == b.read_bytes()
    assert not ({path_group(t) for t in payload['splits']['test']} & {'Kite/trajectory_0003', 'Kite/trajectory_0021'})
    assert all(len(t) % 2 == 0 for t in payload['splits'].values())
    assert load_or_create_manifest(records, a, existing=a) == payload
    altered = json.loads(a.read_text())
    trajectory = altered['splits']['train'].pop()
    altered['splits']['val'].append(trajectory)
    with pytest.raises(ValueError, match='group leakage'):
        validate_manifest(altered, records)
    altered = json.loads(a.read_text())
    pilot = [t for t in sum(altered['splits'].values(), []) if path_group(t) == 'Kite/trajectory_0003']
    old_test = altered['splits']['test']
    for split in ('train', 'val'):
        if set(pilot) <= set(altered['splits'][split]):
            altered['splits'][split] = [t for t in altered['splits'][split] if t not in pilot] + old_test
    altered['splits']['test'] = pilot
    with pytest.raises(ValueError, match='pilot path'):
        validate_manifest(altered, records)


def test_hdf5_wrong_weather_path_rejected(weather_dataset):
    with h5py.File(weather_dataset / 'Kite_training/sunny/sensor_records.hdf5', 'r+') as handle:
        paths = handle['trajectory_0003/camera_data/depth']
        paths[0] = paths[0].decode().replace('sunny', 'foggy')
    with pytest.raises(ValueError, match='synchronization'):
        validate_dataset(weather_dataset)


def test_bundle_setup_and_notebook_syntax(tmp_path):
    packager = module('packager', ROOT / 'scripts/package_kaggle.py')
    setup = module('setup', ROOT / 'scripts/kaggle_setup.py')
    archive = tmp_path / 'code.zip'
    result = packager.package(archive)
    assert result['bytes'] < 200_000
    staged = setup.stage_project(archive, tmp_path / 'staged', result['sha256'])
    assert (staged / 'pyproject.toml').exists()
    with zipfile.ZipFile(archive) as zipped:
        assert not any(n.startswith(('.git/', 'datasets/', '.venv/')) or n.endswith(('.pt', '.hdf5')) for n in zipped.namelist())
    extracted = setup.stage_project(staged, tmp_path / 'extracted')
    assert (extracted / 'src/aerodepth_lite/model.py').read_bytes() == (ROOT / 'src/aerodepth_lite/model.py').read_bytes()
    with pytest.raises(ValueError, match='checksum'):
        setup.extract_project(archive, tmp_path / 'bad', 'wrong')
    notebook = json.loads((ROOT / 'notebooks/kaggle_train.ipynb').read_text())
    assert notebook['nbformat'] == 4
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            ast.parse(source)
            assert cell['outputs'] == []
            assert '"test"' not in source and "'test'" not in source
    report = setup.environment_report(tmp_path)
    assert report['storage_free_gb'] > 0


def test_unsafe_bundle_rejected(tmp_path):
    setup = module('unsafe_setup', ROOT / 'scripts/kaggle_setup.py')
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('../escape.py', 'pass')
    with pytest.raises(ValueError, match='Unsafe'):
        setup.extract_project(archive, tmp_path / 'out')


def test_climate_digit_grouping_and_environment():
    assert {path_group(f'Kite_training/{weather}/trajectory_{number}') for weather, number in
        [('sunny', '0003'), ('sunset', '1003'), ('foggy', '2003'), ('cloudy', '3003')]} == {'Kite/trajectory_0003'}
    assert path_group('PLE_training/winter/trajectory_2021') == 'PLE/trajectory_0021'
    assert path_group('extra/foggy/trajectory_2021') == 'unknown/trajectory_0021'
