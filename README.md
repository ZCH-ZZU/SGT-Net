# SGT-Net

Code repository for

**SGT-Net: A Three-Stage Framework for Cross-Subject Hybrid EEG–fNIRS Decoding
via Temporal Synchronization and Parameter-Efficient Personalization**

---

## Repository layout

```
SGT-Net/
├── README.md                 % this file
├── matlab/                   % preprocessing: raw data -> HDF5
│   ├── README.md             % MATLAB documentation
│   ├── run_preprocessing.m   % single entry point
│   └── +sgtnet/              % implementation package
├── data/                     % raw datasets
│   ├── public/               % benchmark data
│   └── stroke/               % clinical data
├── python/                   % training / evaluation
└── datasets/                 % generated HDF5 files
```

The expected raw-data layout is documented in [`data/README.md`](data/README.md).
The MATLAB and Python parts are independent: MATLAB produces self-describing
HDF5 files, and the Python code reads those files. The raw datasets are not
redistributed here; download them from the original providers (see below).

---

## Quick start

### 1. Preprocess the data (MATLAB)

Requirements: MATLAB R2021b+ and the Signal Processing Toolbox.

```matlab
cd matlab

% Both datasets, using the default data locations
run_preprocessing('all', 'Overwrite', true)

% Or with explicit paths
run_preprocessing('public', ...
    'PublicRoot', 'D:/data/hybridBCI/data', ...
    'OutputDir',  'D:/data/SGT-Net', ...
    'Overwrite',  true)

run_preprocessing('stroke', ...
    'StrokeRoot', 'D:/data/HEFMI-ICH/epo_data', ...
    'OutputDir',  'D:/data/SGT-Net', ...
    'Overwrite',  true)
```

This produces

```
hybridBCI_Tn8.h5            % 29 healthy subjects, MI and MA tasks
stroke_patients_Tn8.h5      % 20 patients, MI task
```

See [`matlab/README.md`](matlab/README.md) for the raw-data layout, the
preprocessing protocol and the HDF5 schema.

### 2. Train and evaluate (Python)

The Python implementation is in `python/`; see
[`python/README.md`](python/README.md) for details. The three entry points are:

```bash
cd python
python main_Stage1.py --config config/Stage1_public.yaml
python main_Stage2.py --config config/Stage2_public.yaml
python main_Stage3.py --config config/Stage3_public.yaml
```

Use the `*_stroke.yaml` variants for the clinical cohort. The expected input is
the pair of HDF5 files above; every subject group contains `eeg_data`,
`hbo_data`, `hbr_data`, `labels_decimal` and `trial_ids`.

---

## Datasets

### Benchmark: Open Access EEG+NIRS Dataset

* Paper: Shin et al., *Open Access Dataset for EEG+NIRS Single-Trial
  Classification*, IEEE TNSRE, 2017.
* Download: <http://doc.ml.tu-berlin.de/hBCI>
* Content: 29 healthy subjects, simultaneous EEG (30 channels + 2 EOG, 200 Hz)
  and fNIRS (36 channels, 10 Hz), motor imagery and mental arithmetic tasks.
* Preprocessing: raw optical intensities are converted with the modified
  Beer–Lambert law; EEG and fNIRS are filtered, epoched and windowed.

### Clinical: HEFMI-ICH

* Paper: Shi et al., *HEFMI-ICH: A Hybrid EEG-fNIRS Motor Imagery Dataset for
  Brain-Computer Interface in Intracerebral Hemorrhage*, Scientific Data, 2025.
* Download: <https://doi.org/10.6084/m9.figshare.28955456.v4>
* Content: 20 patients with intracerebral hemorrhage and unilateral motor
  deficits, 32 EEG channels (256 Hz) and 90 fNIRS channels (11 Hz).
* Preprocessing: the release provides pre-epoched trials and HbO/HbR
  concentration changes; the pipeline resamples to 200/10 Hz, filters,
  crops the common analysis window and applies the same windowing protocol.

---

## Preprocessing protocol at a glance

| | EEG | fNIRS |
| --- | --- | --- |
| Sampling rate | 200 Hz | 10 Hz |
| Band-pass | 4–35 Hz, 6th-order Butterworth, zero-phase | 0.01–0.1 Hz |
| Reference | common average | — |
| Epoch | −5 to 20 s | −5 to 20 s |
| Baseline | −3 to 0 s | −3 to 0 s |
| Window | 2 s, starts 1–8 s, step 1 s | 8 s, starts 1–8 s, step 1 s |
| Normalization | z-score | z-score |

---

## Generated HDF5 files

Both files follow the same schema:

```
/<TASK>/subject_XX/
    eeg_data        [time, channels, samples]
    hbo_data        [time, channels, samples]
    hbr_data        [time, channels, samples]
    labels_decimal  [1, samples]                    (1-based)
    trial_ids       [1, samples]                    (1-based)
```

`trial_ids` identifies the parent trial of every sliding window.

Expected sizes:

| File | Tasks | Subjects | Samples per subject |
| --- | --- | --- | --- |
| `hybridBCI_Tn8.h5` | MI, MA | 29 | 480 per task (60 trials x 8) |
| `stroke_patients_Tn8.h5` | MI | 20 | 240–720 (30–90 trials x 8) |

---

## Citation

If you use this repository, please cite the SGT-Net manuscript and the two
dataset papers:

```bibtex
@article{shin2017,
  author  = {Shin, Jaeyoung and von L{\"u}hmann, Alexander and Blankertz, Benjamin
             and Kim, Do-Won and Jeong, Jichai and Hwang, Han-Jeong
             and M{\"u}ller, Klaus-Robert},
  title   = {Open Access Dataset for {EEG+NIRS} Single-Trial Classification},
  journal = {IEEE Trans. Neural Syst. Rehabil. Eng.},
  year    = {2017},
  volume  = {25},
  number  = {10},
  pages   = {1735--1745},
  doi     = {10.1109/TNSRE.2016.2628057}
}

@article{shi2025,
  author  = {Shi, Jian and Chen, Danyang and Zhao, Xingwei and Zhao, Zhixian
             and Li, Shengjie and Xu, Yeguang and Ding, Tao and Zhu, Zheng
             and Zhang, Peng and Ye, Qing and others},
  title   = {{HEFMI-ICH}: A Hybrid {EEG-fNIRS} Motor Imagery Dataset for
             Brain-Computer Interface in Intracerebral Hemorrhage},
  journal = {Scientific Data},
  year    = {2025},
  volume  = {12},
  number  = {1},
  pages   = {1816},
  doi     = {10.1038/s41597-025-06100-7}
}
```

## License

MIT License. See [`LICENSE`](LICENSE). The two datasets remain under the terms
of their original providers.
