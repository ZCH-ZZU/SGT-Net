# Raw data

This folder is the default location for the two raw datasets. It is not
tracked by Git (only this README and the empty folder placeholders are).
The preprocessing pipeline only reads these files; it never modifies them.

```
data/
├── public/
│   ├── EEG/
│   │   └── subject 01/
│   │       └── with occular artifact/
│   │           ├── cnt.mat
│   │           ├── mrk.mat
│   │           └── mnt.mat
│   └── NIRS/
│       └── subject 01/
│           ├── cnt.mat
│           ├── mrk.mat
│           └── mnt.mat
└── stroke/
    ├── Patients/
    │   ├── EEG/subject1/1_epo.mat, 2_epo.mat, ...
    │   └── NIRS/subject1/1_epo.mat, 2_epo.mat, ...
    └── Normals/
        └── ...
```

## What to copy where

### `data/public`

Copy the **contents** of `<hybrid-BCI-master>/data` here, so that the paths are

```
data/public/EEG/subject 01/with occular artifact/cnt.mat
data/public/NIRS/subject 01/cnt.mat
```

Do not copy the `hybrid-BCI-master` folder itself; otherwise the pipeline
would look for `data/public/hybrid-BCI-master/data/EEG/...`.

Download: <http://doc.ml.tu-berlin.de/hBCI>

`data/physio_metadata.csv` contains the physiological metadata of the public
healthy cohort and is used only by the optional Stage II physio-guided source
selection. It does not apply to the stroke cohort.

### `data/stroke`

Copy the **contents** of `<HEFMI-ICH>/epo_data` here, so that the paths are

```
data/stroke/Patients/EEG/subject1/1_epo.mat
data/stroke/Patients/NIRS/subject1/1_epo.mat
```

Do not copy the `28955456` or `epo_data` folder itself. `Normals/` may stay
but is not used by the default clinical pipeline.

Download: <https://doi.org/10.6084/m9.figshare.28955456.v4>

## Run

With the layout above, no path arguments are needed:

```matlab
cd ../matlab
run_preprocessing('all', 'Overwrite', true)
```

If the raw data is stored elsewhere, pass the two roots explicitly:

```matlab
run_preprocessing('all', ...
    'PublicRoot', 'D:/data/hybridBCI/data', ...
    'StrokeRoot', 'D:/data/HEFMI-ICH/epo_data', ...
    'OutputDir',  'D:/data/SGT-Net', ...
    'Overwrite',  true)
```

`PublicRoot` must be the folder that contains `EEG/` and `NIRS/`.
`StrokeRoot` must be the folder that contains `Patients/` (and optionally
`Normals/`). See `../matlab/README.md` for details.

The raw data occupy several GB. Make sure there is enough free disk space
before running the preprocessing.
