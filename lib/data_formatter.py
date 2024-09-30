"""Organizes and formats the data for easier use"""

import numpy as np
import pandas as pd
import statistics as stat
import math

#Section 1: reading from csvs to df and ndarray

mat_list = ['rubber', 'wood']
f_type_list = ['static', 'kinetic']
color_list = ['darkorange', 'lightorange', 'hotpink', 'yellowneon', 'green']

def trial_to_df(material : str, f_type : str, n : int) -> pd.DataFrame:
    """Takes terms and int as input and returns the corresponding data"""
    df = pd.read_csv(f"Data/friction_{material}_{f_type}_{n}.csv")
    return df

def trialdf_to_nda(df : pd.DataFrame, color : str) -> np.ndarray(2,):
    """Takes a dataframe from a trial and isolates the position of a single dot into an array"""
    pos_dat = np.array([df[f'position_px_x-{color}'], df[f'position_px_y-{color}']])
    return pos_dat

#Section 2: 