import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import logging
import warnings

from neuralforecast import NeuralForecast
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic, generate_series
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.pytorch import DistributionLoss, GMM
from neuralforecast.models import TSMixerx, DilatedRNN, TimeLLM, PatchTST, TCN, TFT, TimesNet, TiDE, iTransformer
from neuralforecast.auto import AutoPatchTST, AutoiTransformer
from sklearn.preprocessing import MinMaxScaler
from neuralforecast.losses.pytorch import MSE
from ray import tune

import os
os.environ['NIXTLA_ID_AS_COL'] = 'True'


##########

pred_per = np.load("CamelsUS_FitPredictions.npy")
yin_per = np.load("CamelsUS_yin.npy")

l, m ,n = yin_per.shape
input_p = pd.read_csv('processed_data.csv')
locs = list((set(input_p["basin_id"])))
dates = list(input_p["Year_Mnth_Day"][21:7031])
dates_all =dates*len(locs)

yin_per_new = yin_per.transpose(1,0,2).reshape(l*m,n)

num_Dates = 7031 - 21
uniques = []
for i in locs:
    uniques+= [i]*num_Dates
    
multi_variables = {
    "precipitation": ["solar radiation", "min_temp", "max_temp", "vapor_pressure"],
    "min_temp": ["precipitation", "solar radiation",  "vapor_pressure"],
    "max_temp": ["precipitation", "solar radiation",  "vapor_pressure"],
    "streamflow": ["precipitation", "solar radiation", "min_temp", "max_temp", "vapor_pressure"],
    "vapor_pressure": ["solar radiation", "min_temp", "max_temp", "precipitation"],
    "solar_radiation":[]

}

import pandas as pd
from datasetsforecast.long_horizon import LongHorizon
from datasetsforecast.m4 import M4
    

variate = "max_temp"
#model_name = "DilatedRNN"
model_name = "PatchTST"

def model_execution(model_name,variate):
    
    
    ##data, X_df, S_df = LongHorizon.load(directory='./data', group='TrafficL')
    data, _,  S_df = M4.load(directory='./data', group='Daily')

    # Function to apply Min-Max normalization
    def min_max_normalize(group):
        min_y = group['y'].min()
        max_y = group['y'].max()
        group['y'] = (group['y'] - min_y) / (max_y - min_y)
        # date_range = pd.date_range(start='200-01-01', periods=len(group), freq='2W')
        # group['ds'] = date_range
        group['ds'] = pd.to_datetime(group['ds'])
        return group

    # Calculate the size of each group
    group_sizes = data.groupby('unique_id').size()
    # Get the top 500 unique_id with the maximum length
    top_500_unique_ids = group_sizes.nlargest(534).index
    # Filter the original DataFrame to keep only the rows belonging to these top 500 groups
    filtered_df = data[data['unique_id'].isin(top_500_unique_ids)]
    # Apply normalization group by group
    Pre_train_data = filtered_df.groupby('unique_id').apply(min_max_normalize)
    
    
    
    df = pd.DataFrame(yin_per_new, columns=["precipitation", "solar_radiation", "min_temp", "max_temp", "vapor_pressure", "streamflow"])
    for col in ["precipitation", "solar_radiation", "min_temp", "max_temp", "vapor_pressure", "streamflow"]:
        df[col].fillna(df[col].mean(), inplace=True)

    df["ds"] = dates_all
    df['ds'] = pd.to_datetime(df['ds'])
    df['unique_id'] = uniques
    df = df.rename(columns={ variate: "y"})
    print(df)

    all_locs = 671
    locs = 534
    val_locs = 137


    df_train = df[:(locs)*7010]
    Y_df_train = df_train
    
    dt_test = df[(val_locs)*7010:]
    Y_df_test = dt_test
    print(df)

    #############


    n_series = locs
    ##ff_dim = 32
    ff_dim = 16
    n_block = 2 #default 4
    print(ff_dim,n_block)


    if(model_name == "TSMixerx"):
#This block represents the TSMixer model
        model = TSMixerx(
                        h=1,
                        input_size=21,
                        n_series=n_series,
                        #stat_exog_list=['basin_id'],
                        hist_exog_list=multi_variables[variate],
                        n_block=n_block,
                        ff_dim=ff_dim,
                       ## dropout=0.8,
                        revin=True,
                        scaler_type="standard",
                        max_steps=1000,  ##1000,
                        early_stop_patience_steps=-1,
                        val_check_steps=100,  ##100,
                        learning_rate=1e-3,
                        loss=MAE(),
                        valid_loss=MAE(),
                        batch_size=32,
        )
    elif(model_name == "DilatedRNN"):
        model = DilatedRNN(
                    h=1,
                    input_size=21,
                   # n_series=n_series,
                    #stat_exog_list=['basin_id'],
                    encoder_hidden_size=200,
                    context_size=10,
                    decoder_hidden_size=200,
                    decoder_layers=2,
                   ## dropout=0.8,
                    scaler_type="standard",
                    max_steps=1000,
                    early_stop_patience_steps=-1,
                    val_check_steps=100,
                    learning_rate=1e-3,
                    loss=MAE(),
                    valid_loss=MAE(),
                    batch_size=32,)
    elif(model_name == "TimeLLM"):
        prompt_prefix = "The dataset contains data on daily hydrology. There is a yearly seasonality"
        model = TimeLLM(h=1,
                     input_size=21,
                    ## llm='openai-community/gpt2',
                   ##  llm_tokenizer='openai-community/gpt2',
                     prompt_prefix=prompt_prefix,
                     batch_size=32,
                     valid_batch_size=16,
                     windows_batch_size=16)
    elif(model_name == "PatchTST"):
#         nhits_config = {
#    ##    "hist_exog_list": multi_variables[variate],
#        "patch_len": tune.choice([8,16,24]), 
#        "stride": tune.choice([4,8,12]), 
#        "learning_rate": tune.loguniform(1e-5, 1e-1),  
#        "hidden_size": tune.choice([8,16,24]), 
#        "n_heads": tune.choice([2,4,8]),
#         "input_size": 21,
#         "revin": False,
#         "scaler_type": 'robust',
#         "max_steps": 500,
#         "val_check_steps": 50,
            
#     }
#         model = AutoPatchTST(
#                  h=1,
#                  config=nhits_config,
#                  loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
#                  )
        model = PatchTST(h = 1, input_size=21, max_steps=32, learning_rate=1e-4, loss= MSE(), random_seed=1, step_size=1, batch_size=256, scaler_type=None)
        fcst = NeuralForecast(models=[model], freq="D")

        for i in range(100):
            print("fit :", i+1)
            fcst.fit(Pre_train_data, verbose=False, val_size=0)


        for i in range(10):
            print("fit :", i+1)
            fcst.fit(Y_df_train, verbose=False, val_size=0)
        
    elif(model_name == "TCN"):
        model = TCN(h=1,
                input_size=-21,
                #loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                loss=GMM(n_components=7, return_params=True, level=[80,90]),
                learning_rate=5e-4,
                kernel_size=2,
                dilations=[1,2,4,8,16],
                encoder_hidden_size=128,
                context_size=10,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=500,
                scaler_type='robust',
                   )
        
        
    elif(model_name == "TFT"):
        model = TFT(h=1, 
                input_size=21,
                hidden_size=20,
                learning_rate=0.005,
                hist_exog_list=multi_variables[variate],
                max_steps=500,
                val_check_steps=50,
                scaler_type='robust',
                loss=MSE(),
                valid_loss=MAE(),
                )
    elif(model_name == "TimesNet"):
        model = TimesNet(h=1,
                 input_size=21,
                 hidden_size = 16,
                 conv_hidden_size = 32,
                 loss=MSE(),
                 valid_loss=MAE(),
                 scaler_type='standard',
                 learning_rate=1e-3,
                 max_steps=500,
                 val_check_steps=50,)
    elif(model_name == "iTransformer"):
#         nhits_config = {
#    ##    "hist_exog_list": multi_variables[variate],
#        "hidden_size": tune.choice([8,16,24]), 
#        "n_heads": tune.choice([2,4,8]),
#         "input_size": 21,
#         "n_series": n_series,
#         "scaler_type": 'robust',
#         "max_steps": 500,
#         "val_check_steps": 50,
#         "hidden_size":tune.choice([64,128,256]) ,
#         "e_layers": tune.choice([4,2,8]),
#         "d_layers": tune.choice([2,1,4]),
#         "d_ff": tune.choice([8,4,16]),
#         "batch_size":32,

#     }
#         model = AutoiTransformer(
#                  h=1,
#                  config=nhits_config,
#                  loss=MSE(),
#                  valid_loss=MAE(),
#                  n_series= n_series
#                  )
        horizon = 1
        input_size = 21
        model = iTransformer(h = horizon, input_size=input_size, n_series=n_series, max_steps=32, learning_rate=1e-4, loss= MSE(), random_seed=10, step_size=1, batch_size=256, scaler_type=None)
        fcst = NeuralForecast(models=[model], freq='D', local_scaler_type=None)

        #PRE-TRAINING
        for i in range(40):
            print("fit :", i+1)
            fcst.fit(Pre_train_data, verbose=True, val_size=0)


#FINE-TUNING
        for i in range(4):
            print("fit :", i+1)
            fcst.fit(Y_df_train, verbose=True, val_size=0)
        
    elif(model_name == "TiDE"):
        model = TiDE(h=1,
                input_size=21,
                loss=MSE(),
                valid_loss=MAE(),
                max_steps=500,
                val_check_steps= 50,
                scaler_type='standard',)    



   # fcst = NeuralForecast(models=[model], freq="D")

   # fcst.fit(df=Y_df_train)

    print(len(Y_df_train))
    ##########
    grouped_df = Y_df_train.groupby("unique_id")



    length =7010
    forcasted = []
    forcasted_test = []
    MAE_val, MSE_val = 0,0
    n = 0
    for i in range(30,length):
        combined_df = pd.concat([group[i-21:i] for _, group in grouped_df])
        considered = pd.concat([group[i:i+1] for _, group in grouped_df])
        forecasts = fcst.predict(df=combined_df,futr_df=fcst.make_future_dataframe(df=combined_df))
        MAE_val += sum(abs(x - y) for x, y in zip(considered["y"], forecasts[model_name]))
        MSE_val += sum(abs(x - y)**2 for x, y in zip(considered["y"], forecasts[model_name]))
        forcasted.append(forecasts)
        n+=1
    MAE_val = round(MAE_val/(n*locs),5)
    MSE_val = round(MSE_val/(n*locs),5)

    forcasted = pd.concat(forcasted)
    

    train_pred_plt = [0]*31 +list(forcasted.groupby(['ds']).sum()[model_name])
    train_orgn_plt = list(Y_df_train.groupby(['ds']).sum()["y"])[:length]
    train_residual = list(map(lambda x, y: x - y, train_orgn_plt, train_pred_plt))

    grouped_df = Y_df_test.groupby("unique_id")

    forcasted_test = []
    for i in range(30,length):
        combined_df = pd.concat([group[i-21:i] for _, group in grouped_df])
        forecasts = fcst.predict(df=combined_df,futr_df=fcst.make_future_dataframe(df=combined_df))
        forcasted_test.append(forecasts[(locs - val_locs):])

    forcasted_test = pd.concat(forcasted_test)
    
    
    test_pred_plt = [0]*31 +list(forcasted_test.groupby(['ds']).sum()[model_name])
    test_orgn_plt = list(Y_df_test.groupby(['ds']).sum()["y"])[:length]
    test_residual = list(map(lambda x, y: x - y, test_orgn_plt, test_pred_plt))


    ######

    import pickle
    file = "results/"+model_name + "_" + variate + '.pkl'
    with open(file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([train_pred_plt, train_orgn_plt, train_residual, test_pred_plt, test_orgn_plt, test_residual], f)

    print("Values MSE,MAE",MSE_val,MAE_val)
    return "Values MSE,MAE" + str(MSE_val) + "   " + str(MAE_val)
    
import atexit
vals = []
def cleanup_function():
    # Code to execute before termination
    print(vals)
    # Perform any necessary cleanup tasks here

atexit.register(cleanup_function)    
    
    
##["precipitation", "solar_radiation", "min_temp", "max_temp", "vapor_pressure", "streamflow"]
for model_name in ["iTransformer"]:
    for variate in ["min_temp", "max_temp", "solar_radiation"]:
        results= model_execution(model_name,variate)
        vals.append(variate + " M4 " +results)
        with open("2.txt", "a") as f:
            f.write(model_name +" "+ variate + "  " +results+ "\n")
print(vals)