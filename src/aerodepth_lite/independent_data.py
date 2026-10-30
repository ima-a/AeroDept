"""Metadata-only preparation of the four pre-reserved independent test paths."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
from collections import Counter
import argparse, gzip, hashlib, json, os, shutil, stat, zipfile
from urllib.parse import urlsplit, unquote
import h5py
import numpy as np
from .inference import file_sha256

ROOT = Path(__file__).resolve().parents[2]
RESERVATIONS = ROOT/'results/expanded_preparation_20261026/reservations.json'
TRAIN_VAL = ROOT/'results/expanded_preparation_20261026/manifest_train_val.json.gz'
EXPECTED = {'Kite_training/foggy/trajectory_2007':2204,
            'Kite_training/sunny/trajectory_0008':2205,
            'PLE_training/spring/trajectory_5008':2206,
            'PLE_training/winter/trajectory_6009':2205}
STREAMS = {'color_left':'JPEG', 'depth':'PNG'}

def write_new(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def group(tr):
    env,_,name=tr.split('/')
    return env.split('_')[0]+'/'+f'{int(name.split("_")[-1])%1000:04d}'

# Archives
def safe_members(z, count, suffix):
    rows=z.infolist();names=[r.filename for r in rows]
    expected={f'{i:06d}.{suffix}' for i in range(count)}
    if len(rows)!=count or len(set(names))!=count or set(names)!=expected:
        raise ValueError('Archive filenames/count do not match the complete reserved stream')
    for r in rows:
        mode=r.external_attr>>16
        if r.is_dir() or r.flag_bits&1 or PurePosixPath(r.filename).name!=r.filename or stat.S_ISLNK(mode) or r.file_size<=0:
            raise ValueError('Unsafe archive member')
    return sorted(rows,key=lambda r:r.filename)

# Leakage
def verify_isolation(test_positions, reservations, manifest):
    if set(reservations['new_test_realizations'])!=set(EXPECTED):raise ValueError('Reservations changed')
    tests={group(t) for t in EXPECTED}
    exposed=set(manifest['path_groups']) | {group(t) for t in reservations['legacy_test_realizations']} | set(reservations['pilot_excluded_from_new_test'])
    if tests & exposed:raise ValueError('Reserved paths overlap prior exposure')
    if len(test_positions)!=4 or len(set(test_positions.values()))!=4:raise ValueError('Test position streams are not distinct')
    known=reservations['position_identity_audit']
    for tr,digest in test_positions.items():
        matching=[x for x in known if x['trajectory']==tr]
        if len(matching)!=1 or matching[0]['position_sha256']!=digest:raise ValueError('Reserved sensor position identity changed')
        if any(x['position_sha256']==digest and x['group']!=group(tr) for x in known):
            raise ValueError('Identical position stream under another path identity')
        if any(x['position_sha256']==digest and x['group'] in exposed for x in known):
            raise ValueError('Position stream overlaps exposed path')
    return {'passed':True,'reserved_groups':sorted(tests),'exposed_groups':sorted(exposed),
            'weather_equivalents_checked':len(known),'position_sha256':test_positions,
            'limitation':'Path isolation does not establish geographic/scenery isolation.'}

def sensor_records(sensor_root):
    reservations=json.loads(RESERVATIONS.read_text())
    manifest=json.loads(gzip.decompress(TRAIN_VAL.read_bytes()))
    records=[];positions={};sensors={};details={}
    for tr,count in sorted(EXPECTED.items()):
        parent,name=tr.rsplit('/',1);path=sensor_root/parent/'sensor_records.hdf5'
        if not path.is_file() or path.is_symlink():raise ValueError('Missing existing sensor record')
        sensor_rel=parent+'/sensor_records.hdf5';sensors[sensor_rel]=file_sha256(path)
        with h5py.File(path,'r') as f:
            g=f[name];positions[tr]=hashlib.sha256(np.asarray(g['groundtruth/position'][:]).tobytes()).hexdigest()
            images={s:[v.decode() if isinstance(v,bytes) else str(v) for v in g['camera_data/'+s][:]] for s in STREAMS}
            if any(len(v)!=count for v in images.values()):raise ValueError('HDF5 camera count differs')
            streams={n:(np.asarray(g[n][:]),float(g[n].attrs['sampling_frequency'])) for n in ['imu/accelerometer','imu/gyroscope','gps/velocity']}
            if any(not np.isfinite(v).all() or v.ndim!=2 or v.shape[1]!=3 for v,hz in streams.values()):raise ValueError('Invalid navigation')
            if [streams[n][1] for n in streams]!=[100.,100.,1.]:raise ValueError('Unexpected sensor rates')
            details[tr]={'frames':count,'camera_hz':25,'imu_hz':100,'gps_hz':1,'position_sha256':positions[tr],
                         'alignment':'floor(camera_ordinal*sensor_hz/25); held sample; no interpolation or lookahead'}
            for i in range(count):
                nav={}
                for n,keys in [('imu/accelerometer',('ax','ay','az')),('imu/gyroscope',('wx','wy','wz')),('gps/velocity',('gps_vx','gps_vy','gps_vz'))]:
                    values,hz=streams[n];idx=int(i*hz/25)
                    if idx>=len(values):raise ValueError('Sensor stream does not cover camera timestamps')
                    nav.update(zip(keys,map(float,values[idx])))
                row={'trajectory_id':tr,'frame_index':i,'timestamp':i/25,'navigation':nav}
                for s,ext in STREAMS.items():
                    expected=f'{s}/{name}/{i:06d}.{ext}'
                    if images[s][i]!=expected:raise ValueError('HDF5 filename correspondence differs')
                    row['rgb_path' if s=='color_left' else 'depth_path']=parent+'/'+expected
                records.append(row)
    return records,sensors,details,verify_isolation(positions,reservations,manifest)

def inventory(downloads, sensor_root, destination, output):
    downloads=downloads.resolve();sensor_root=sensor_root.resolve();destination=destination.absolute()
    if destination.exists():raise FileExistsError('Staging destination must be new')
    if destination==ROOT or ROOT in destination.parents:raise ValueError('Dataset staging must be outside repository')
    records,sensors,details,isolation=sensor_records(sensor_root)
    cfg=downloads/'download_config.txt';canonical=[]
    for line in cfg.read_text().splitlines():
        url=urlsplit(line.strip());parts=unquote(url.path).split('/')
        hits=[i for i,p in enumerate(parts) if p in ('Kite_training','PLE_training')]
        if url.hostname=='midair.ulg.ac.be' and len(hits)==1:canonical.append('/'.join(parts[hits[0]:]))
    rows=[]
    for tr,count in sorted(EXPECTED.items()):
        parent,name=tr.rsplit('/',1)
        for stream,ext in STREAMS.items():
            relative=f'{parent}/{stream}/{name}/frames.zip';p=downloads/'MidAir'/relative
            if canonical.count(relative)!=1 or not p.is_file() or p.is_symlink():raise ValueError('Acquisition identity cannot be resolved: '+relative)
            with zipfile.ZipFile(p) as z:
                members=safe_members(z,count,ext)
                rows.append({'path':'MidAir/'+relative,'trajectory':tr,'stream':stream,'frames':count,
                             'compressed_bytes':p.stat().st_size,'uncompressed_bytes':sum(x.file_size for x in members),
                             'sha256':file_sha256(p)})
    partials=[str(p.relative_to(downloads)) for p in (downloads/'MidAir').rglob('*') if p.is_file() and p.suffix.lower() in ('.part','.crdownload','.aria2','.tmp')]
    if partials:raise ValueError('Partial study downloads: '+str(partials))
    source_bytes=sum((sensor_root/s).stat().st_size for s in sensors)
    extracted=sum(x['uncompressed_bytes'] for x in rows)
    ancestor=destination.parent
    while not ancestor.exists():ancestor=ancestor.parent
    free=shutil.disk_usage(ancestor).free;reserve=2*1024**3;outputs=512*1024**2
    if free<extracted+source_bytes+reserve+outputs:raise ValueError('Insufficient healthy staging capacity')
    other=[str(p.relative_to(downloads)) for p in downloads.rglob('*.zip') if str(p.relative_to(downloads)) not in {r['path'] for r in rows}]
    result={'status':'metadata_verified_not_exposed','archives':rows,'download_config_sha256':file_sha256(cfg),
            'sensor_sha256':sensors,'sensor_correspondence':details,'leakage':isolation,'expected_frames':sum(EXPECTED.values()),
            'reservations_sha256':file_sha256(RESERVATIONS),'train_val_manifest_sha256':file_sha256(TRAIN_VAL),
            'storage':{'compressed_bytes':sum(x['compressed_bytes'] for x in rows),'extracted_image_bytes':extracted,'sensor_copy_bytes':source_bytes,
                       'free_before':free,'output_allowance':outputs,'minimum_free_reserve':reserve,'enough':True},
            'unrelated_or_redundant_archives':other,'partial_files':partials,
            'identity_resolution':'Hierarchy and official download-config paths agree; no scene appearance used.',
            'privacy':'Download URLs and credentials omitted; config digest only.'}
    write_new(output,result);return result

# Extraction
def extract(downloads,sensor_root,destination,inventory_path,out):
    info=json.loads(inventory_path.read_text());out.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError('Never overwrite a staging tree')
    if info['status']!='metadata_verified_not_exposed':raise ValueError('Inventory not ready')
    if file_sha256(RESERVATIONS)!=info['reservations_sha256'] or file_sha256(TRAIN_VAL)!=info['train_val_manifest_sha256']:raise ValueError('Reference split changed')
    records,sensors,details,isolation=sensor_records(sensor_root)
    if sensors!=info['sensor_sha256'] or isolation!=info['leakage']:raise ValueError('Sensor/isolation identity changed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    need=info['storage']['extracted_image_bytes']+info['storage']['sensor_copy_bytes']+info['storage']['output_allowance']+info['storage']['minimum_free_reserve']
    if shutil.disk_usage(destination.parent).free<need:raise ValueError('Staging space changed')
    destination.mkdir();hashes={};verified=[]
    for row in info['archives']:
        p=downloads/row['path']
        if file_sha256(p)!=row['sha256']:raise ValueError('Archive changed before extraction')
        tr=row['trajectory'];parent,name=tr.rsplit('/',1);folder=destination/parent/row['stream']/name;folder.mkdir(parents=True)
        with zipfile.ZipFile(p) as z:
            for member in safe_members(z,EXPECTED[tr],STREAMS[row['stream']]):
                target=folder/member.filename;h=hashlib.sha256();size=0
                with z.open(member) as source,target.open('xb') as dest:
                    for block in iter(lambda:source.read(1024*1024),b''):h.update(block);dest.write(block);size+=len(block)
                if size!=member.file_size:raise ValueError('Extracted size differs')
                hashes[target.relative_to(destination).as_posix()]={'sha256':h.hexdigest(),'bytes':size}
        verified.append(row['path']);print('CRC and extraction verified:',row['trajectory'],row['stream'],flush=True)
    for relative,digest in sensors.items():
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(sensor_root/relative,target)
        if file_sha256(target)!=digest:raise ValueError('Sensor copy mismatch')
    for r in records:
        for field in ['rgb','depth']:
            details=hashes[r[field+'_path']];r[field+'_sha256']=details['sha256'];r[field+'_bytes']=details['bytes']
    manifest={'version':'reserved_independent_test_v1','frame_counts':{'test':sum(EXPECTED.values())},'per_trajectory':EXPECTED,
              'sensor_sha256':sensors,'records':records,'navigation_fields':'Causal simulated IMU and held GPS only; no GT velocity.',
              'reservations_sha256':info['reservations_sha256'],'train_val_manifest_sha256':info['train_val_manifest_sha256']}
    target=out/'manifest.json.gz'
    with target.open('xb') as f:f.write(gzip.compress(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    write_new(out/'PREPARED.json',{'status':'verified_not_exposed','frames':len(records),'image_files':len(hashes),'archives_crc_verified':verified,
              'manifest_sha256':file_sha256(target),'inventory_sha256':file_sha256(inventory_path),'sensor_sha256':sensors,
              'leakage':isolation,'decoder_access':False,'free_after':shutil.disk_usage(destination).free})
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['inventory','extract']);p.add_argument('--downloads',type=Path,required=True)
    p.add_argument('--sensor-root',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    if a.stage=='inventory':r=inventory(a.downloads,a.sensor_root,a.destination,a.inventory);print(json.dumps(r['storage']))
    else:
        if not a.output:p.error('--output required for extraction')
        r=extract(a.downloads,a.sensor_root,a.destination,a.inventory,a.output);print('Prepared',r['frame_counts'])

if __name__=='__main__':main()
