"""Organizes and formats the data for easier use"""
#This whole library is just a bunch of numpy and pandas stuff to manipulate arrays
# so that I don't have to write these a bunch later on

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

#Section 1: reading from csvs to df and ndarray

mat_list = ['rubber', 'wood']
f_type_list = ['static', 'kinetic']

color_list = ['darkorange', 'lightorange', 'hotpink', 'green', 'yellowneon']

def trial_to_df(material : str, f_type : str, n : int) -> pd.DataFrame:
    """Takes terms and int as input and returns the corresponding data"""
    df = pd.read_csv(f"bin/friction_{material}_{f_type}_{n}.csv")
    df = df.dropna(axis='index', how='any')
    return df

def trialdf_to_nda(df : pd.DataFrame, color : str) -> np.ndarray(shape=(2,)):
    """Takes a dataframe from a trial and isolates the position of a single dot into an array"""
    pos_dat = np.array([df[f'position_px_x-{color}'], df[f'position_px_y-{color}']])
    return pos_dat

#Section 2: Stacking

def a_stack(a : np.ndarray(shape=(1,)), b : np.ndarray(shape=(1,))) -> np.ndarray(shape=(2,)):
    """stacks two 1d numpy arrays"""
    return np.stack(a, b)

def a_unstack(a : np.ndarray(shape=(2,))) -> (np.ndarray(shape=(1,)), np.ndarray(shape=(1,))):
    """unstacks a 2d numpy array into two 1d arrays"""
    return a[0], a[1]

def l_stack(a : list, b: list) -> np.ndarray(shape=(2,)):
    """Stacks two lists into a 2d array"""
    return np.stack((np.array(a), np.array(b)))

def l_unstack(a, axis=0) -> (list, list):
    """takes a 2d array and returns the rows as lists in order"""
    return np.moveaxis(a, axis, 0)

#Section 3: stacking part 2, 3d boogaloo

def a_3d_stack(a):
    """Stacks 2D numpy arrays on top of eachother"""
    return np.array(a)



def a_3d_slice(A, axis = 2):
    """Takes a 3d array and returns a list of it's slices along an axis (0-2)"""
    shape = A.shape                                 # shape of the array

    if axis == 2:                                   # along the 2 axis
        slices_2 = []
        l_2 = shape[2]                              # length along the 2 axis
        for i in range(l_2):
            slice_2 = A[:, :, i]                    # takes a slice
            slice_2_t = slice_2.T                   # formats it properly
            slices_2.append(np.array(slice_2_t))    # adds it to list of slices
        return slices_2
    
    elif axis == 1:                                 # along the 1 axis
        slices_1 = []
        l_1 = shape[1]                              # length along the 1 axis
        for i in range(l_1):
            slice_1 = A[:, i, :]                    # takes a slice
            slices_1.append(np.array(slice_1))      # adds it to list of slices
        return slices_1
    
    elif axis == 0:                                 # along the 0 axis
        slices_0 = []
        l_0 = shape[0]                              # length along 0 axis
        for i in range(l_0):
            slice_0 = A[i, :, :]                    # takes a slice
            slices_0.append(np.array(slice_0))      # adds it to list of slices
        return slices_0
    
    else:
        raise ValueError(f"Axis {axis} is out of bounds for a 3D array.")



def a_3d_recombine(A, axis = 2):
    """Takes a list of slices and recombines them along a specified axis (0-2)"""

    if axis == 2:                                   # along the 2 axis
        slices_2_t = [slice_2.T for slice_2 in A]   # transposes slices
        recom_2 = np.stack(slices_2_t, axis = 2)    # recombines along 2-axis
        return recom_2
    
    elif axis == 1:                                 # along the 1 axis
        recom_1 = np.stack(A, axis = 1)             # recombines
        return recom_1
    
    elif axis == 0:                                 # along the 0 axis
        recom_0 = np.stack(A, axis = 0)             # recombines
        return recom_0

    else:
        raise ValueError(f"Axis {axis} is out of bounds for a 3D array.")
    
#Section 4: putting the data together into a 3d array

def trial_to_3d(material : str, f_type : str, n : int):
    """Takes in parameters for a given trial and returns a 3d matrix of the positions"""
    df = trial_to_df(material, f_type, n)
    stacks_2d = []
    for color in color_list:
        stacks_2d.append(np.array(trialdf_to_nda(df, color)))
    out = a_3d_stack(stacks_2d)
    return out

# Other stuff

def smooth_data(data, window_length=5, polyorder=2):
    """Applies Savitzky-Golay filter to smooth data"""
    if len(data) < window_length:
        raise ValueError("Data length is less than window length.")
    return savgol_filter(data, window_length, polyorder)



def fdiff(x : list, y : list, use_smoothing = True):
    """Applies finite difference to a data set"""
    if len(x)!=len(y):
        raise ValueError("unaligned sets for fdiff")
    
    if use_smoothing:
        x_smooth = np.array(x)
        y_smooth = smooth_data(np.array(y))
    else:
        x_smooth = np.array(x)
        y_smooth = np.array(y)

    x_prime = []
    y_prime = []

    for i in range (1, len(x_smooth)-1):
        dx = x_smooth[i+1]-x_smooth[i-1]
        dy = y_smooth[i+1]-y_smooth[i-1]
        m = dy/dx
        x_prime.append(float(x_smooth[i]))
        y_prime.append(float(m))
    return x_prime, y_prime