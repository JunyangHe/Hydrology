# Deep Learning Foundations and Pattern Models: Challenges in Hydrological Time Series

Junyang He, Ying-Jung Chen, Anushka Idamekorala, Geoffrey Fox

This repo contains the LSTM model and data preprocessing code for the paper [Deep Learning Foundations and Pattern Models: Challenges in Hydrological Time Series](https://arxiv.org/abs/2410.15218).


## Hydrology-LSTM.ipynb
Our executable model script is designed to be run on Google Colab. 

### Train
- Training checkpoints will be stored in a directory named "checkpoints" in the specified root directory.
- Locations used for training and validation are stored in a file named "Validation[RunName]" in the specified root directory.

### Evaluation
- One must select a previously finished run to evaluate.
- Plots will be stored in a directory named "Outputs" in the specified root directory.


## Data
Available for download from [Zenodo](https://doi.org/10.5281/zenodo.13975174).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13975174.svg)](https://doi.org/10.5281/zenodo.13975174)

### Data Structure
```
hydrology_data_processed
├── CAMELS-US
│   ├── BasicInputStaticProps.npy
│   ├── BasicInputTimeSeries.npy
│   └── metadata.json
│
├── CAMELS-combined
│   ├── us
│   │   ├── BasicInputStaticProps_us_combined.npy
│   │   ├── BasicInputTimeSeries_us_combined.npy
│   │   └── metadata_us_combined.json
│   ├── gb
│   │   ├── BasicInputStaticProps_gb_combined.npy
│   │   ├── BasicInputTimeSeries_gb_combined.npy
│   │   └── metadata_gb_combined.json
│   └── cl
│       ├── BasicInputStaticProps_cl_combined.npy
│       ├── BasicInputTimeSeries_cl_combined.npy
│       └── metadata_cl_combined.json
│
└── Caravan
    ├── camels
    │   ├── BasicInputStaticProps_camels.npy
    │   ├── BasicInputTimeSeries_camels.npy
    │   └── metadata_camels.json
    ├── camelsaus
    │   ├── BasicInputStaticProps_camelsaus.npy
    │   ├── BasicInputTimeSeries_camelsaus.npy
    │   └── metadata_camelsaus.json
    ├── camelsbr
    │   ├── BasicInputStaticProps_camelsbr.npy
    │   ├── BasicInputTimeSeries_camelsbr.npy
    │   └── metadata_camelsbr.json
    ├── camelscl
    │   ├── BasicInputStaticProps_camelscl.npy
    │   ├── BasicInputTimeSeries_camelscl.npy
    │   └── metadata_camelscl.json
    ├── camelsgb
    │   ├── BasicInputStaticProps_camelsgb.npy
    │   ├── BasicInputTimeSeries_camelsgb.npy
    │   └── metadata_camelsgb.json
    ├── hysets
    │   ├── BasicInputStaticProps_hysets.npy
    │   ├── BasicInputTimeSeries_hysets.npy
    │   └── metadata_hysets.json
    └── lamah
        ├── BasicInputStaticProps_lamah.npy
        ├── BasicInputTimeSeries_lamah.npy
        └── metadata_lamah.json

```

## Cite
```
@misc{he2024sciencetimeseriesdeep,
      title={Science Time Series: Deep Learning in Hydrology}, 
      author={Junyang He and Ying-Jung Chen and Anushka Idamekorala and Geoffrey Fox},
      year={2024},
      eprint={2410.15218},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.15218}, 
}
```
