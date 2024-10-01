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

timestamp = df['timestamp'].tolist()

timestamp = [i/1000 for i in timestamp]

full = datform.a_3d_stack([DO, O, HP, G ,YN])
full_norm = trig.normalise_major(full)

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

print(O_HP_unit)
print(shape)

units = []

for i in range(l):
    current_slice = O_HP_unit[:,i]
    current_slice = np.reshape(current_slice, (2, 1))
    units.append(np.array(current_slice))

for i in units:
    current_theta = trig.theta_from_posx(i)
    theta_list.append(float(current_theta))

dx_list = []

for n in range(len(DO_n[0])):
    dx = DO_n[0][n]-O_n[0][n]
    dx_list.append(float(dx))
    
ts_list, v_list = datform.fdiff(x = timestamp, y = dx_list, use_smoothing=True)

plt.plot(theta_list[1:-1], v_list)
plt.title("LO-DO v_x vs angle")
plt.show()

plt.plot(ts_list, theta_list[1:-1])
plt.title("Angle vs time")
plt.show()

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(ts_list, theta_list[1:-1], v_list)
ax.set_xlabel('time')
ax.set_ylabel('angle')
ax.set_zlabel('velocity')
plt.show()