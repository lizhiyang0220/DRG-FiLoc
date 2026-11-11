# DRG-FiLoc

This repository contains the DRG-FiLoc method code and saved localization prediction results for four Wi-Fi fingerprinting datasets: IPIN2016, IPIN2018, Tampere2017, and UJIIndoorLoc.

Training and evaluation data files are included for IPIN2016 and IPIN2018. For Tampere2017 and UJIIndoorLoc, the original datasets must be downloaded separately from the sources listed below. **Prediction results are provided for all four datasets.**

## Repository contents

| Directory | Contents |
| --- | --- |
| `IPIN2016/` | DRG-FiLoc method code, training/evaluation CSV files, and prediction results for CAR, UAH, UJITI, and UJIUB |
| `IPIN2018/` | DRG-FiLoc method code, training/test Excel files, and prediction results |
| `Tampere2017/` | DRG-FiLoc method code and prediction results; original dataset must be downloaded separately |
| `UJIIndoorLoc/` | DRG-FiLoc method code and prediction results; original dataset must be downloaded separately |

## Prediction results

The following files contain the saved localization predictions and per-sample errors. Each row records a ground-truth location, a predicted location, and the corresponding Euclidean localization error.

| Dataset | Prediction file | Format | Samples |
| --- | --- | --- | ---: |
| IPIN2016 – CAR | [`IPIN2016_CAR_predictions.xlsx`](IPIN2016/IPIN2016_CAR_predictions.xlsx) | XLSX | 509 |
| IPIN2016 – UAH | [`IPIN2016_UAH_predictions.xlsx`](IPIN2016/IPIN2016_UAH_predictions.xlsx) | XLSX | 912 |
| IPIN2016 – UJITI | [`IPIN2016_UJITI_predictions.xlsx`](IPIN2016/IPIN2016_UJITI_predictions.xlsx) | XLSX | 304 |
| IPIN2016 – UJIUB | [`IPIN2016_UJIUB_predictions.xlsx`](IPIN2016/IPIN2016_UJIUB_predictions.xlsx) | XLSX | 165 |
| IPIN2018 | [`IPIN2018_predictions.csv`](IPIN2018/IPIN2018_predictions.csv) | CSV | 291 |
| Tampere2017 | [`Tampere_predictions.csv`](Tampere2017/Tampere_predictions.csv) | CSV | 697 |
| UJIIndoorLoc | [`UJIIndoorLoc_predictions.csv`](UJIIndoorLoc/UJIIndoorLoc_predictions.csv) | CSV | 1,111 |

### Prediction file format

**IPIN2016 and IPIN2018** contain the following columns:

| Column | Description |
| --- | --- |
| `fingerprint_index` | Index of the evaluated fingerprint |
| `true_x_m`, `true_y_m` | Ground-truth two-dimensional coordinates, in meters |
| `pred_x_m`, `pred_y_m` | Predicted two-dimensional coordinates, in meters |
| `euclidean_error_m` | Euclidean distance between the ground-truth and predicted coordinates, in meters |

**Tampere2017** uses the same coordinate and error fields, but names its index column `sample_index` instead of `fingerprint_index`.

**UJIIndoorLoc** uses the following columns:

| Column | Description |
| --- | --- |
| `fingerprint_index` | Index of the evaluated fingerprint |
| `true_longitude`, `true_latitude` | Ground-truth coordinate values, using the names stored in the output file |
| `pred_longitude`, `pred_latitude` | Predicted coordinate values, using the names stored in the output file |
| `euclidean_error` | Euclidean distance between the ground-truth and predicted coordinate pairs |

For UJIIndoorLoc, consult the dataset documentation and the method code to interpret the coordinate reference system and units; the column names alone do not establish that the values are geographic degrees.

The prediction files are provided to facilitate result inspection and further analysis. They do not replace the original fingerprint datasets required to rerun the experiments.

## Dataset sources

| Dataset | Download/source |
| --- | --- |
| IPIN2016 | [Zenodo record, DOI: 10.5281/zenodo.2791530](https://doi.org/10.5281/zenodo.2791530) |
| IPIN2018 | [Zenodo record, DOI: 10.5281/zenodo.2823964](https://doi.org/10.5281/zenodo.2823964) |
| Tampere2017 | [Zenodo record, DOI: 10.5281/zenodo.889798](https://doi.org/10.5281/zenodo.889798) |
| UJIIndoorLoc | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/310/ujindoorloc); [official UJI database page](https://indoorloc.uji.es/) |

For Tampere2017, the linked Zenodo record provides the crowdsourced Wi-Fi fingerprinting dataset collected at Tampere University of Technology in 2017. UJIIndoorLoc is available from UCI and the dataset authors' UJI page.

## Included IPIN files

### IPIN2016

`IPIN2016/` includes `IPIN2016_DRG-FiLoc.py`, paired training/evaluation CSV files for CAR, UAH, UJITI, and UJIUB, and the four corresponding prediction Excel files listed above. The supplied test/evaluation files use the `wifi_evaluation_*.csv` naming convention.

### IPIN2018

`IPIN2018/` includes `IPIN2018_DRG-FiLoc.py`, `wifi_training.xlsx`, `wifi_test.xlsx`, and `IPIN2018_predictions.csv`.

## Reproducibility notes

- Download the separately hosted original datasets from the links above and follow their documentation for dataset layout and terms of use.
- The supplied prediction files contain previously generated results; rerunning the method requires the corresponding source data and execution environment.
- The exact Python version and package requirements have not yet been specified in this repository. Add them here, or provide a `requirements.txt` or environment file, to improve reproducibility.
- Consult each original dataset record for its license, citation, and redistribution terms.

## Citation

Please cite the DRG-FiLoc paper associated with this repository and the original dataset records used in your work. Add the paper's complete citation here before publication.
