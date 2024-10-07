import os
import numpy as np
import pandas as pd
import math
from scipy.signal import savgol_filter
import lib.data_formatter as datform
import lib.trig as trig
import lib.phys as phys

"""Main file, intended to be run"""

#stack order: DO, LO, HP, G, YN

#units are m, kg, s

mass = 0.2241   # kg

LO_HP = 0.178   # m
G_YN = 0.217    # m

A_r = 0.00378   # m^2
A_w = 0.00364   # m^2

def px_to_m(A) -> float:
    """Takes in the data for a timestep and returns the number to multiply px distance by to get m distance"""
    LO = A[:, 1].reshape(2, 1)
    HP = A[:, 2].reshape(2, 1)
    G = A[:, 3].reshape(2, 1)
    YN = A[:, 4].reshape(2, 1)
    distLOHP = trig.distance(LO, HP)
    distGYN = trig.distance(G, YN)
    factorLOHP = LO_HP / distLOHP
    factorGYN = G_YN / distGYN
    factor = (factorLOHP + factorGYN)/2
    return factor

def static(material : str):
    """Does all the static friction stuff"""
    test_list = [i for i in os.listdir("bin/") if f"{material}_static" in i]
    length = len(test_list)
    full_coeff_list = []
    std_list = []
    for i in range(1, length + 1):

        #setting up the full data
        full = datform.trial_to_3d(material, "static", i, True)
        full_norm = trig.normalise_major(A = full)
        full_norm_sliced = datform.a_3d_slice(A = full_norm, axis = 2)
        
        #setting up the partial data
        sample = np.array(datform.a_3d_slice(A = full_norm, axis = 0)[0:3])
        sample_sliced = datform.a_3d_slice(A = sample, axis = 2)

        #using the full data to get the conversion factors for each timestep
        factor_list = []
        for j in full_norm_sliced:
            factor_list.append(px_to_m(j))
        factor_list = factor_list[1:-1]

        df = pd.read_csv(f"bin/friction_{material}_static_{i}.csv")
        df = df.dropna(axis='index', how='any')
        timestamp = [i/1000 for i in df['timestamp'].tolist()]

        DO_n = sample[0]
        O_n = sample[1]
        HP_n = sample[2]

        O_HP_unit = np.array([[],[]])

        for j in range(len(O_n[0])):
            dx = HP_n[0][j]-O_n[0][j]
            dy = O_n[1][j]-HP_n[1][j]
            O_HP = np.array([[dx], [dy]])
            l = np.linalg.norm(O_HP)
            O_HP_hat = O_HP / l
            O_HP_unit = np.concatenate((O_HP_unit, O_HP_hat), axis=1)

        theta_list = []
        shape = O_HP_unit.shape
        l = shape[1]
        units = []

        for j in range(l):
            current_slice = O_HP_unit[:,j]
            current_slice = np.reshape(current_slice, (2, 1))
            units.append(np.array(current_slice))

        for j in units:
            current_theta = trig.theta_from_posx(j)
            theta_list.append(float(current_theta))

        dx_list = []

        for j in range(len(DO_n[0])):
            dx = DO_n[0][j]-O_n[0][j]
            dx_list.append(float(dx))
            
        ts_list, v_list = datform.fdiff(x = timestamp, y = dx_list, use_smoothing = True)
        v_list_new = []
        for j in range(len(v_list)):
            factor = factor_list[j]
            v_list_new.append(float(v_list[j]*factor))

        theta_list = theta_list[2:-2]

        tsa_list, a_list = datform.fdiff(x = ts_list, y = v_list_new, use_smoothing = True)

        a_vs_ang = np.array([[list(a_list)],[list(theta_list)]])

        thresh = 0.5

        useful_ang_list = a_vs_ang[1, a_vs_ang[0] > thresh]
        useful_ang = np.mean(useful_ang_list)
        useful_ang_std = np.std(useful_ang_list)
        coeff = phys.calc_fric_s(useful_ang)
        std = (1 / (np.cos(useful_ang)**2)) * useful_ang_std
        full_coeff_list.append(float(coeff))
        std_list.append(float(std))
    std = trig.pythag_inf(std_list)
    mean_coeff = np.mean(full_coeff_list)
    return(mean_coeff, std)
        

def kinetic(material : str):
    """Does all the kinetic friction stuff"""
    test_list = [i for i in os.listdir("bin/") if f"{material}_kinetic" in i]
    length = len(test_list)
    full_coeff_list = []
    std_list =[]
    for i in range(1, length + 1):

        #setting up the full data
        full = datform.trial_to_3d(material, "kinetic", i, True)
        full_norm = trig.normalise_major(A = full)
        full_norm_sliced = datform.a_3d_slice(A = full_norm, axis = 2)
        
        #setting up the partial data
        sample = np.array(datform.a_3d_slice(A = full_norm, axis = 0)[0:3])
        sample_sliced = datform.a_3d_slice(A = sample, axis = 2)

        #using the full data to get the conversion factors for each timestep
        factor_list = []
        for j in full_norm_sliced:
            factor_list.append(px_to_m(j))
        factor_list = factor_list[1:-1]

        #after this i just copied over from the dev.py and it worked after some edits

        df = pd.read_csv(f"bin/friction_{material}_kinetic_{i}.csv")
        df = df.dropna(axis='index', how='any')
        timestamp = [i/1000 for i in df['timestamp'].tolist()]

        DO_n = sample[0]
        O_n = sample[1]
        HP_n = sample[2]

        O_HP_unit = np.array([[],[]])

        for j in range(len(O_n[0])):
            dx = HP_n[0][j]-O_n[0][j]
            dy = O_n[1][j]-HP_n[1][j]
            O_HP = np.array([[dx], [dy]])
            l = np.linalg.norm(O_HP)
            O_HP_hat = O_HP / l
            O_HP_unit = np.concatenate((O_HP_unit, O_HP_hat), axis=1)

        theta_list = []
        shape = O_HP_unit.shape
        l = shape[1]
        units = []

        for j in range(l):
            current_slice = O_HP_unit[:,j]
            current_slice = np.reshape(current_slice, (2, 1))
            units.append(np.array(current_slice))

        for j in units:
            current_theta = trig.theta_from_posx(j)
            theta_list.append(float(current_theta))

        dx_list = []

        for j in range(len(DO_n[0])):
            dx = DO_n[0][j]-O_n[0][j]
            dx_list.append(float(dx))
            
        ts_list, v_list = datform.fdiff(x = timestamp, y = dx_list, use_smoothing = True)
        v_list_new = []
        for j in range(len(v_list)):
            factor = factor_list[j]
            v_list_new.append(float(v_list[j]*factor))

        tsa_list, a_list = datform.fdiff(x = ts_list, y = v_list_new, use_smoothing = True)
        theta_list = theta_list[2:-2]
        coeff_list = []
        for j in range(len(a_list)):
            coeff_list.append(float(phys.calc_fric_k(theta_list[j], a_list[j])))
        
        coeff = np.mean(coeff_list)
        coeff_std = np.std(coeff_list)
        full_coeff_list.append(float(coeff))
        std_list.append(float(coeff_std))

    std = trig.pythag_inf(std_list)
    mean_coeff = np.mean(full_coeff_list)
    return(mean_coeff, std)

def main():
    """Everything all together"""
    r_k, r_k_s = kinetic("rubber")
    w_k, w_k_s = kinetic("wood")
    r_s, r_s_s = static("rubber")
    w_s, w_s_s = static("wood")

    print(f"rubber kinetic: {r_k} +- {r_k_s}")
    print(f"wood kinetic: {w_k} +- {w_k_s}")
    print(f"rubber static: {r_s} +- {r_s_s}")
    print(f"wood static: {w_s} +- {w_s_s}")


if __name__ == "__main__":
    main()