[Uploading README.md…]()
# DRG-FiLoc

This repository contains the method code for four Wi-Fi fingerprinting datasets: IPIN2016, IPIN2018, Tampere2017, and UJIIndoorLoc. Training and evaluation files are included for IPIN2016 and IPIN2018. For Tampere2017 and UJIIndoorLoc, download the original dataset from the sources below; their data files are not included here.

## Repository contents

| Directory | Contents |
| --- | --- |
| `IPIN2016/` | DRG-FiLoc method code and training/evaluation CSV files for CAR, UAH, UJITI, and UJIUB |
| `IPIN2018/` | DRG-FiLoc method code and training/test Excel files |
| `Tampere2017/` | DRG-FiLoc method code; download dataset separately |
| `UJIIndoorLoc/` | DRG-FiLoc method code; download dataset separately |

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

`IPIN2016/` includes `IPIN2016_DRG-FiLoc.py` and paired training/evaluation CSV files for CAR, UAH, UJITI, and UJIUB. The supplied test/evaluation files use the `wifi_evaluation_*.csv` naming convention.

### IPIN2018

`IPIN2018/` includes `IPIN2018_DRG-FiLoc.py`, `wifi_training.xlsx`, and `wifi_test.xlsx`.

## Reproducibility notes

- Download the separately hosted datasets from the links above and follow their original documentation for dataset layout and terms of use.
- The exact Python version and package requirements have not yet been specified in this repository. Add them here (or provide a `requirements.txt`/environment file) to make the method easier to reproduce.
- Dataset files are provided for research reproduction. Please consult each dataset's source record for its license, citation, and redistribution terms.

## Citation

Please cite the DRG-FiLoc paper associated with this repository and cite the original dataset records used in your work. Add the paper's complete citation here before publication.
