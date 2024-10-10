# Science Time Series: Deep Learning in Hydrology

Junyang He, Ying-Jung Chen, Anushka Idamekorala, Geoffrey Fox

## Train
- Training checkpoints will be stored in a directory named "checkpoints" in the "Hydrology" directory.
- Locations used for training and validation are stored in a file named "Validation[RunName]" in the "Hydrology" directory.

## Evaluation
- One must select a previously finished run to evaluate.
- Plots will be stored in a directory named "Outputs" in the "Hydrology" directory.

## Prerequisites
- Google Colab
- python 3
- pandas
- numpy
- Tensorflow v.15.1

## Repository Structure
```
Hydrology
│
├── data
│   ├── CAMELS-US-preprocessed
│   ├── CAMELS-combined-preprocessed
│   ├── Caravan
│   │   ├── 
│   │   ├──
│   │   ├── 
│   │   ├──
│   │   ├── 
│   │   ├──
│   │   └──
├── preprocessing
│   ├── camels_us_preprocess.ipynb
│   ├── camels_gb_preprocess.ipynb
│   ├── camels_cl_preprocess.ipynb
│   └── caravan_preprocess.ipynb
├── Hydrology-LSTM.ipynb
```
