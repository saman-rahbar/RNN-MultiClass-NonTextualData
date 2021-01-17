#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday Jan 17 13:44:56 2020

@author: Sam Rahbar
"""

from keras.layers.core import Activation
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.layers import LeakyReLU
import os
import sys
from scipy import stats
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import logging



LOG_FILENAME = 'logging_training.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    )

RANDOM_SEED = 42
RAW_INPUT_PATH = "/home/sam/Desktop/supervised_rnn/RNN/dataset/supervised_rnn_data.csv"



def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)        
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)



def plot_cm(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16)) 
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap=sns.diverging_palette(220, 20, n=7),
        ax=ax
    )

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()


def plot_state_breakDown(state, df):
    data = df[df['state'] == state][['0', '1', '2','3']][:119]
    axis = data.plot(subplots=True, figsize=(16, 12), 
                    title=state)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

### MODEL PARAMETERS ###
# '[relu, tf.keras.layers.LeakyReLU(alpha=0.2)]'
Activation = tf.keras.layers.LeakyReLU(alpha=0.2)
########################

if __name__ == "__main__":

    if os.path.isdir("../data_reports"):
        pass
    else:
        os.mkdir("../data_reports")

    if os.path.isdir("../train_reports"):
        pass
    else:
        os.mkdir("../train_reports")

    register_matplotlib_converters()
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    rcParams['figure.figsize'] = 22, 10

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Reading the dataset
    # Reading the dataset
    df = pd.read_csv(RAW_INPUT_PATH)
    df.dropna(axis=0, how='any', inplace=True)
    df.head()
    logging.debug("Dataset shape " + str(df.shape))

    # More reports on data
    sns.countplot(x = 'state',
                data = df,
                order = df.state.value_counts().index)
    
    plt.savefig("../data_reports/Records_per_state.png")
    plt.title("Records per state")
    plt.close()

    sns.countplot(x = 'user_id',
                data = df,
                palette=[sns.color_palette()[0]],
                order = df.user_id.value_counts().index)

    plt.savefig("../data_reports/Records_per_user.png")
    plt.title("Records per user")
    plt.close()
    

    plot_state_breakDown("A", df)
    plt.savefig("../data_reports/A-breakdown.png")
    plt.close()
    plot_state_breakDown("B", df)
    plt.savefig("../data_reports/B-breakdown.png")
    plt.close()
    plot_state_breakDown("C", df)
    plt.savefig("../data_reports/C-breakdown.png")
    plt.close()
    plot_state_breakDown("D", df)
    plt.savefig("../data_reports/D-breakdown.png")
    plt.close()

    df_train = df[df['user_id'] <= 30]
    df_test = df[df['user_id'] > 30]

    scale_columns = ['0', '1', '2', '3']

    scaler = RobustScaler()

    scaler = scaler.fit(df_train[scale_columns])

    df_train.loc[:, scale_columns] = scaler.transform(df_train[scale_columns].to_numpy())
    df_test.loc[:, scale_columns] = scaler.transform(df_test[scale_columns].to_numpy())

    TIME_STEPS = 119
    STEP = 40

    X_train, y_train = create_dataset(
        df_train[['0', '1', '2', '3']], 
        df_train.state, 
        TIME_STEPS, 
        STEP
    )

    X_test, y_test = create_dataset(
        df_test[['0', '1', '2', '3']], 
        df_test.state, 
        TIME_STEPS, 
        STEP
    )

    logging.debug(X_train.shape, y_train.shape)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    logging.debug(X_train.shape, y_train.shape)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128, 
            input_shape=[X_train.shape[1], X_train.shape[2]]
        )
        )
    )
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation=Activation))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=1000,
    validation_split=0.1,
    shuffle=True
    )

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')


    plt.legend()
    plt.savefig("../train_reports/train_val_loss.png")
    plt.close()
    
    model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    plot_cm(
    enc.inverse_transform(y_test),
    enc.inverse_transform(y_pred),
    enc.categories_[0]
    )
    plt.savefig("../train_reports/confusion_matrix.png")
    plt.close()
    f = open(LOG_FILENAME, 'rt')
    try:
        body = f.read()
    finally:
        f.close()