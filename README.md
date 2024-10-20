# Science Time Series: Deep Learning in Hydrology

Junyang He, Ying-Jung Chen, Anushka Idamekorala, Geoffrey Fox

## Paper
arXiv: 

## Train
- Training checkpoints will be stored in a directory named "checkpoints" in the "Hydrology" directory.
- Locations used for training and validation are stored in a file named "Validation[RunName]" in the "Hydrology" directory.

## Evaluation
- One must select a previously finished run to evaluate.
- Plots will be stored in a directory named "Outputs" in the "Hydrology" directory.

## Data Structure
```
data
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
