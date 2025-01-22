from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import pandas as pd
import matplotlib
import math
import scipy.sparse as sparse
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
import networkx as nx
import spektral
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from spektral.layers.convolutional import GCNConv, GATConv
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from keras.losses import Loss, categorical_crossentropy
from keras.metrics import binary_accuracy
from keras.losses import CategoricalCrossentropy, MeanSquaredError
from keras.optimizers import Adam
from spektral.data.graph import Graph
from spektral.data.dataset import Dataset
from spektral.data.loaders import SingleLoader, MixedLoader
from sklearn.utils import compute_class_weight, compute_sample_weight
from keras import Model, Sequential
import keras as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import os
import warnings
from tqdm import tqdm

variates = ["max_temp","vapor_pressure","streamflow"]
for variate in variates:
    n_timesteps_in_sample =21
    df = pd.read_csv("data_points.csv",index_col=False)
    df = df.rename(columns={ variate: "y"})#
    temp_df = df.drop('Unnamed: 0', axis=1)#
    temp_df = df[["ds","y","unique_id"]]#
    #temp_df= df
    #temp_df = df.drop('Unnamed: 0', axis=1)

    n_time_series = 671
    Adj = pd.read_csv('adj_matrix_1.csv', index_col=0)

    print('network loaded.')
    class MyDataset(Dataset):
        def __init__(self, a, **kwargs):
            super().__init__(**kwargs)
            self.a = a
        def read(self):
            return list_of_Graphs

    # Group by 'unique_id'
    grouped = temp_df.groupby('unique_id')
    # Get the first unique_id and its corresponding 'ds' values
    df_timesteps = grouped.get_group(grouped['ds'].first().index[0])['ds']

    list_of_Graphs = []
    for i in tqdm(df_timesteps[n_timesteps_in_sample:], desc="Processing timesteps"):
        daily_graph_x = []
        daily_graph_y = []
        for _, group in grouped:
            filtered_group = group[group['ds'] < i]
            if len(filtered_group) >= n_timesteps_in_sample:
                X_day = filtered_group.iloc[-n_timesteps_in_sample:]
                Y_day = group[group['ds'] == i]
                if not Y_day.empty:
                    daily_graph_x.append(X_day[['y']].values.flatten())
                    daily_graph_y.append(Y_day[['y']].values.flatten())           
        if daily_graph_x and daily_graph_y:
            daily_graph_x = pd.DataFrame(daily_graph_x)
            daily_graph_y = pd.DataFrame(daily_graph_y)
            Graph = spektral.data.graph.Graph(x=daily_graph_x.astype('float32').values, y=daily_graph_y.astype('float32').values)
            list_of_Graphs.append(Graph)

    set_of_graphs = MyDataset(a=Adj.values)
    set_of_graphs.read()
    n = 2000


    print('number of total graphs : ',len(set_of_graphs))
    train_graphs = set_of_graphs[:-n]
    test_graphs = set_of_graphs[-n:]

    print(set_of_graphs)
    print(train_graphs)
    print(test_graphs)
    import pickle
    with open('set_of_grpahs_'+variate+'_1.pkl', 'wb') as f:
        pickle.dump(set_of_graphs, f)

