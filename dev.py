import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.data_formatter as datform
import lib.trig as trig
from scipy.signal import savgol_filter

mat_list = ['rubber', 'wood']
type_list = ['static', 'kinetic']
color_list = ['darkorange', 'lightorange', 'hotpink', 'yellowneon', 'green']
plot_color_list = ['darkorange', 'orange', 'pink', 'yellow', 'green']

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

def a_by_angle(material : str, f_type : str, n : int):
    df = pd.read_csv(f"bin/friction_{material}_{f_type}_{n}.csv")
    df = df.dropna(axis='index', how='any')
    DO = np.array([df['position_px_x-darkorange'].tolist(), df['position_px_y-darkorange'].tolist()])
    O = np.array([df['position_px_x-lightorange'].tolist(), df['position_px_y-lightorange'].tolist()])
    HP = np.array([df['position_px_x-hotpink'].tolist(), df['position_px_y-hotpink'].tolist()])
    G = np.array([df['position_px_x-green'].tolist(), df['position_px_y-green'].tolist()])
    YN = np.array([df['position_px_x-yellowneon'].tolist(), df['position_px_y-yellowneon'].tolist()])

    timestamp = df['timestamp'].tolist()

    timestamp = [i/1000 for i in timestamp]

    full = datform.a_3d_stack([DO, O, HP, G ,YN])
    full_norm = trig.normalise_major(full)
    full_norm_sliced = (datform.a_3d_slice(A=full_norm, axis=2))

    factor_list = []
    for j in full_norm_sliced:
        factor_list.append(px_to_m(j))
    factor_list = factor_list[1:-1]

    sample = np.array(datform.a_3d_slice(A=full_norm, axis=0)[0:3])

    DO_n = sample[0]
    O_n = sample[1]
    HP_n = sample[2]

    O_HP_unit = np.array([[],[]])

    for i in range(len(O_n[0])):
        dx = HP_n[0][i]-O_n[0][i]
        dy = O_n[1][i]-HP_n[1][i]
        O_HP = np.array([[dx], [dy]])
        l = np.linalg.norm(O_HP)
        O_HP_hat = O_HP / l
        O_HP_unit = np.concatenate((O_HP_unit, O_HP_hat), axis=1)

    theta_list = []
    shape = O_HP_unit.shape
    l = shape[1]
    units = []

    for i in range(l):
        current_slice = O_HP_unit[:,i]
        current_slice = np.reshape(current_slice, (2, 1))
        units.append(np.array(current_slice))

    for i in units:
        current_theta = trig.theta_from_posx(i)
        theta_list.append(float(current_theta))

    dx_list = []

    for i in range(len(DO_n[0])):
        dx = DO_n[0][i]-O_n[0][i]
        dx_list.append(float(dx))
        
    ts_list, v_list = datform.fdiff(x = timestamp, y = dx_list, use_smoothing = True)

    v_list_new = []
    for j in range(len(v_list)):
        factor = factor_list[j]
        v_list_new.append(float(v_list[j]*factor))

    tsa_list, a_list = datform.fdiff(x = ts_list, y = v_list_new, use_smoothing = True)

    plt.plot(theta_list[2:-2], a_list, label=n)

for n in range(1,11):
    a_by_angle("rubber", "static", n)

plt.title("Rubber Static Acceleration vs. Angle")
plt.legend()
plt.show()