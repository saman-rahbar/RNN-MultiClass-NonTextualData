#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday Jan 17 13:44:56 2020

@author: Sam Rahbar
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import sys
from pandas.core.indexes.base import Index


LOG_FILENAME = 'logging_dataset.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    )

CSV_ABS_PATH = "/home/sam/Desktop/supervised_rnn/RNN/dataset/supervised_rnn_data.csv"

def data_preprocesser(df_path, header=None):
    """
    Method to process the data

    Parameters
    ----------
    def_path: str
        the absolute path to the csv file
    header: str, optional
        A flag to read the header of the dataset (defult
        is None)

    Returns
    -------
    Pandas Dataframe
    """

    if header==None:
        dataset= pd.read_csv(df_path)
        dataset.dropna(axis=0, how='any', inplace=True)
        logging.debug("successfully read the raw csv file with header")
    else:
        dataset= pd.read_csv(df_path, header)
        dataset.dropna(axis=0, how='any', inplace=True)
        logging.debug("Successfully read the raw csv file with no header")

    logging.debug("+++++++++++++++++++++++++++++++++")
    logging.debug("+++++ Dataset General Stats +++++")
    logging.debug("+++++++++++++++++++++++++++++++++")
    logging.debug(dataset.describe())
    # Grouping data by users
    data_grouped = dataset.groupby(['user_id'])
    # User id distributions
    data_grouped.count().plot()
    plt.savefig("../data_reports/user_id_counts.png")
    logging.debug("successfully saved the user_id distributions")
    dataset.groupby(["state"]).count().plot()
    logging.debug(dataset.groupby(["state"]).count())
    plt.savefig("../data_reports/states_counts.png")
    logging.debug("successfully saved the state distributions")

    # Encoding states
    cleanup_nums = {"state": {"A": 1, "B": 2, "C": 3, "D": 4}}
    dataset = dataset.replace(cleanup_nums)
    logging.debug("item encoding for the states successfully done!")

    # Saving the processed dataset
    dataset.to_csv("../dataset/supervised_rnn_data_encoded.csv", index=False)
    logging.debug("new dataset saved successfully!")

    print(dataset.head())
    return dataset


if __name__=="__main__":
    if os.path.isdir("../data_reports"):
        pass
    else:
        os.mkdir("../data_reports")
        
    data_preprocesser(CSV_ABS_PATH)
    f = open(LOG_FILENAME, 'rt')
    try:
        body = f.read()
    finally:
        f.close()