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

print('libraries imported .... ')

n_timesteps_in_sample = 21
filterd_df = pd.read_csv('filterd_df.csv', index_col=0)
print('filterd_df loaded .... ')

n_time_series = 671


print('network loaded.........................................................')
class MyDataset(Dataset):
    def __init__(self, a, **kwargs):
        super().__init__(**kwargs)
        self.a = a
    def read(self):
        return list_of_Graphs





class MY_GNN(tf.keras.Model):

    def __init__(
            self,
            n_labels = 1,
            activation="relu",
            output_activation="softmax",
            use_bias=True,
            dropout_rate=0.25,
            l2_reg=1e-5,
            n_input_channels=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        reg = tf.keras.regularizers.l2(l2_reg)


        self.MLP1 = tf.keras.layers.Dense(32, activation=activation, use_bias=use_bias, kernel_initializer="glorot_uniform",
                                    bias_initializer="zeros", kernel_regularizer=reg)

        self.GTA1 = GATConv(8, attn_heads=4, concat_heads=True, dropout_rate=dropout_rate, activation=activation,
                                    kernel_regularizer=reg, use_bias=use_bias)

#         self.GTA2 = GATConv(8, attn_heads=4, concat_heads=True, dropout_rate=dropout_rate, activation=activation,
#                                     kernel_regularizer=reg, use_bias=use_bias)
        
#         self.GTA3 = GATConv(8, attn_heads=4, concat_heads=True, dropout_rate=dropout_rate, activation=activation,
#                                     kernel_regularizer=reg, use_bias=use_bias)
        
        self.MLP2 = tf.keras.layers.Dense(16, activation=activation, use_bias=use_bias, kernel_initializer="glorot_uniform",
                                    bias_initializer="zeros", kernel_regularizer=reg)

        self.final_MLP = tf.keras.layers.Dense(n_labels, activation=None, use_bias=False)

        if tf.version.VERSION < "2.2":
            if n_input_channels is None:
                raise ValueError("n_input_channels required for tf < 2.2")
            x = tf.keras.Input((n_input_channels,), dtype=tf.float32)
            a = tf.keras.Input((None,), dtype=tf.float32, sparse=True)
            self._set_inputs((x, a))


    def get_config(self):
        return dict(
            n_labels=self.n_labels,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
        )

    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs  # So that the model can be used with DisjointLoader


        x1 = self.MLP1(x)
        x2 = self.GTA1([x1, a])
        # x3 = self.GTA2([x2, a])
        # x4 = self.GTA3([x3, a])
        
        concatenated = tf.concat([x1, x2], axis=-1)
        x = self.MLP2(concatenated)

        return self.final_MLP(x)
    
    
    
    
    
################################



# Training step


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def normalized_nash_sutcliffe_efficiencySTavg(y_true, y_pred): # axis 0 space 1 time
    NSE = 1 - np.sum (np.square(y_true - y_pred) ) / np.sum( np.square(y_true - np.mean(y_true)) )
    return 1 / ( 2 - NSE)
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

###################

import pickle
def gnn_run(variate): 
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def normalized_nash_sutcliffe_efficiencySTavg(y_true, y_pred): # axis 0 space 1 time
        NSE = 1 - tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 / ( 2 - NSE)

    
    @tf.function
    def evaluate(loader):
        output = []
        step = 0
        all_preds = []
        all_targets = []

        for inputs, target in loader:
            #target = target[:,:,feature:feature+1]
            step += 1
            pred = model(inputs, training=False)
            loss = loss_fn(target, pred)
            output.append(loss)

            all_preds.append(pred)
            all_targets.append(target)

            if step >= loader.steps_per_epoch:
                break

        all_preds = tf.concat(all_preds, axis=0)
        all_targets = tf.concat(all_targets, axis=0)
        all_preds = tf.squeeze(all_preds, axis=2)
        all_preds = tf.transpose(all_preds, perm=[1, 0]) 
        all_targets = tf.squeeze(all_targets, axis=2)
        all_targets = tf.transpose(all_targets, perm=[1, 0]) 
        NNES = normalized_nash_sutcliffe_efficiencySTavg(all_targets, all_preds)

        return output, all_preds, NNES
    
    
    with open('set_of_grpahs_'+variate+'_3`.pkl', 'rb') as f:
        set_of_graphs = pickle.load(f)

    n = 2000
    train_graphs = set_of_graphs[:6000]
    test_graphs = set_of_graphs[6000:6900]

    print(set_of_graphs)
    print(train_graphs)
    print(test_graphs)

    #-------------------------------------------------------------
    seed = 10
    # Set the random seed for Python's random module
    random.seed(seed)
    # Set the random seed for NumPy
    np.random.seed(seed)
    # Set the random seed for TensorFlow
    tf.random.set_seed(seed)
    print('seed :',seed)
    #-------------------------------------------------------------

    feature = 2
    loader_train = MixedLoader(train_graphs, batch_size=32, shuffle=False, epochs=30)
    loader_test =  MixedLoader(test_graphs, batch_size=32, shuffle=False)

    # Create model
    model = MY_GNN(n_input_channels=set_of_graphs.n_node_features)
    optimizer = Adam(5e-4)
    loss_fn = MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error")




    epoch = step = 0
    train_losses = []
    test_losses = []
    for batch in loader_train:
        step += 1
        inputs, target = batch
       # target = target[:,:,feature:feature+1]
       ## print("1" , inputs[0].shape, inputs[1].shape)
       ## print("To train...")
        loss_train = train_step(inputs, target)
        train_losses.append(loss_train)

        if step == loader_train.steps_per_epoch:
            step = 0
            epoch += 1
            loss_test, pred, nnse = evaluate(loader_test)
            test_losses.append(loss_test)
            inputs, target = loader_test.__next__()
            # Compute the average test loss for better readability
            avg_test_loss = tf.reduce_mean(loss_test)
            print(f"Ep. {epoch} --- Loss: {loss_train:.3f} --- Test loss: {avg_test_loss:.3f} --- Test NNSE: {nnse:.4f}")



    results_te, predictions, nnse = evaluate(loader_test)
    # Convert the TensorFlow tensor to a NumPy array
    predictions = predictions.numpy()

    true_labels_test = np.hstack([one_graph_labels.y for one_graph_labels in test_graphs])




    Y_ACTUAL = true_labels_test
    Y_HAT = predictions

    MAE = mean_absolute_error(Y_ACTUAL, Y_HAT)
    MSE = mean_squared_error(Y_ACTUAL, Y_HAT)
   ## NNSE = normalized_nash_sutcliffe_efficiencySTavg(Y_ACTUAL, Y_HAT)
    results = f" MSE: {MSE} MAE:{MAE} "
    with open("1.txt", "a") as f:
        f.write(" GNN 1 "+ variate + "  "  + results + "\n")

    print("MAE: ", MAE)
    print("MSE: ", MSE)
    #print("NNSE: ", NNSE1)
    return


for variate in ["solar_radiation"]:
    gnn_run(variate)

    
        

            

