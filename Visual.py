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

df = pd.read_csv("bin/friction_rubber_static_3.csv")
df = df.dropna(axis='index', how='any')
DO = np.array([df['position_px_x-darkorange'].tolist(), df['position_px_y-darkorange'].tolist()])
O = np.array([df['position_px_x-lightorange'].tolist(), df['position_px_y-lightorange'].tolist()])
HP = np.array([df['position_px_x-hotpink'].tolist(), df['position_px_y-hotpink'].tolist()])
G = np.array([df['position_px_x-green'].tolist(), df['position_px_y-green'].tolist()])
YN = np.array([df['position_px_x-yellowneon'].tolist(), df['position_px_y-yellowneon'].tolist()])

test_set = datform.a_3d_stack([DO, O, HP])
test_set_norm = trig.normalise_major(test_set)

#set_plot(test_set, "test")
#set_plot(test_set_norm, "test normal")

timestamp = df['timestamp'].tolist()

timestamp = [i/1000 for i in timestamp]

DO_n = test_set_norm[0]
DO_n_x = DO_n[0]
O_n = test_set_norm[1]
O_n_x = O_n[0]

dx_list = []

for n in range(len(DO_n_x)):
    dx = DO_n_x[n]-O_n_x[n]
    dx_list.append(float(dx))

plt.plot(timestamp, dx_list)
plt.title("LO-DO dx")
plt.axhline(color='red', linestyle="--")
plt.show()
    
ts_list, v_list = datform.fdiff(x = timestamp, y = dx_list, use_smoothing=True)

plt.plot(ts_list, v_list)
plt.title("LO-DO v_x")
plt.show()

for n in range(1, 5):
    df = pd.read_csv(f"bin/friction_rubber_static_{n}.csv")
    df = df.dropna(axis='index', how='any')
    DO = np.array([df['position_px_x-darkorange'].tolist(), df['position_px_y-darkorange'].tolist()])
    O = np.array([df['position_px_x-lightorange'].tolist(), df['position_px_y-lightorange'].tolist()])
    HP = np.array([df['position_px_x-hotpink'].tolist(), df['position_px_y-hotpink'].tolist()])
    current_set = datform.a_3d_stack([DO, O, HP])
    current_set_norm = trig.normalise_major(current_set)
    timestamp = df['timestamp'].tolist()
    timestamp = [i/1000 for i in timestamp]
    DO_n_x = current_set_norm[0][0]
    O_n_x = current_set_norm[1][0]
    dx_list = []
    for i in range(len(DO_n_x)):
        dx = DO_n_x[i]-O_n_x[i]
        dx_list.append(float(dx))
    ts_list, v_list = datform.fdiff(x = timestamp, y = dx_list, use_smoothing=True)
    plt.plot(list(ts_list), list(v_list), label = str(n))
plt.legend()
plt.show()

