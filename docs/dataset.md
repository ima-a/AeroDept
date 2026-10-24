# MidAir acquisition and split identity

Use the [official acquisition form](https://midair.ulg.ac.be/download.html) to select left RGB and depth/range, plus the corresponding HDF5 records, and accept the [MidAir terms](https://midair.ulg.ac.be/index.html). The site supplies a download-link file. From a new dataset directory:

```bash
wget --content-disposition -x -nH -i /path/to/download_config.txt
# Extract each selected ZIP alongside itself, preserving its internal directory layout.
```

No data is downloaded by model inference. No raw images, depth arrays, archives or sensor HDF5 files are distributed in this repository. Research results require the same selected data. Published scientific figures are validation-derived and attributed under NOTICE.md; they are not final-test demos.

The frozen manifest lists all 12 actual environment/climate/trajectory IDs and relative RGB/depth paths, grouped splits and resampled original navigation. SHA-256 of the UNCOMPRESSED manifest is `e7b093be69f1942f281a09563bd63efa479d4352de97da722fb0504208c1fbf5`. Its gzip is deterministic and public. HDF5 IDs can differ from the website's base sunny path IDs; select by the published manifest and account for climate prefixes (e.g. website Kite0003 corresponds to foggy2003).

Validation needs only:

```text
$MIDAIR_ROOT/
  Kite_training/foggy/
    color_left/trajectory_2003/000000.JPEG ...
    depth/trajectory_2003/000000.PNG ...
    sensor_records.hdf5
  PLE_training/spring/
    color_left/trajectory_5000/000000.JPEG ...
    depth/trajectory_5000/000000.PNG ...
    sensor_records.hdf5
```

Each validation trajectory has 2,205 synchronized frames. The original expanded subset has 26,481 pairs: 17,635 train, 4,410 validation and 4,436 test. This is a custom grouped split, not the official MidAir benchmark. The test trajectory IDs exist in manifest metadata to freeze the split; this task and the public reproduction runner never load their RGB/depth. Training reproduction also needs the eight training trajectories listed in that manifest. An optional generic discovery integration test is not part of the protected validation protocol and should not be enabled against a reserved final-test extraction.

V1 consumes original manifest navigation: privileged groundtruth velocity and simulated IMU. V2 explicitly reads only selected validation groups' `gps/velocity`, holds sample `floor(camera_ordinal*GPS_frequency/25)`, and combines it with manifest simulated accelerometer/gyroscope values. Effective1Hz GPS and100Hz IMU are aligned to25Hz camera time. Values are simulation proxies with ideal availability/common time origin; physical sensor latency and navigation estimation are untested.
