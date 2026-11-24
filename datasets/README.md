# Generated datasets

This folder receives the HDF5 files produced by `matlab/run_preprocessing.m`:

| File | Content |
| --- | --- |
| `hybridBCI_Tn8.h5` | 29 healthy subjects, MI and MA tasks |
| `stroke_patients_Tn8.h5` | 20 patients, MI task |

The files are several GB in total and are intentionally excluded from version
control (see `../.gitignore`). Regenerate them from the public raw datasets:

* Benchmark: <http://doc.ml.tu-berlin.de/hBCI>
* Clinical: <https://doi.org/10.6084/m9.figshare.28955456.v4>

See `../matlab/README.md` for the raw-data layout and the exact commands.
