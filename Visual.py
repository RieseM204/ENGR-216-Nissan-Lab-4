import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.data_formatter as datform
import lib.trig as trig

mat_list = ['rubber', 'wood']
type_list = ['static', 'kinetic']
color_list = ['darkorange', 'lightorange', 'hotpink', 'yellowneon', 'green']
plot_color_list = ['darkorange', 'orange', 'pink', 'yellow', 'green']

def plot_sep(A : str, B : str, n : int):
    df = pd.read_csv(f'bin/friction_{A}_{B}_{n}.csv')
    for i in range(5):
        pos_x = df[f'position_px_x-{color_list[i]}'].tolist()[3:]
        pos_y = df[f'position_px_y-{color_list[i]}'].tolist()[3:]
        plt.plot(pos_x, pos_y, label=i, color=plot_color_list[i])
    plt.gca().invert_yaxis()
    plt.title(f"{A} {B} {n}")
    plt.show()


plot_sep("rubber", "static", 2)

df = pd.read_csv(f'Data/friction_rubber_static_2.csv')
A_1_x = df[f'position_px_x-lightorange'].tolist()[3:]
A_1_y = df[f'position_px_y-lightorange'].tolist()[3:]

A_2_x = df[f'position_px_x-hotpink'].tolist()[3:]
A_2_y = df[f'position_px_y-hotpink'].tolist()[3:]

B_1_x = df[f'position_px_x-green'].dropna(how="any").tolist()[3:]
B_1_y = df[f'position_px_y-green'].dropna(how="any").tolist()[3:]

B_2_x = df[f'position_px_x-yellowneon'].dropna(how="any").tolist()[3:]
B_2_y = df[f'position_px_y-yellowneon'].dropna(how="any").tolist()[3:]

A_1 = np.array([[max(A_1_x)], [min(A_1_y)]])
A_2 = np.array([[max(A_2_x)], [min(A_2_y)]])

B_1 = np.array([[np.mean(B_1_x)],[np.mean(B_1_y)]])
B_2 = np.array([[np.mean(B_2_x)],[np.mean(B_2_y)]])

print(B_1)

A = trig.origin_unit(A_1, A_2)
A_theta = trig.theta_from_posx(A)

B = trig.origin_unit(B_1, B_2)
B_theta = trig.theta_from_posx(B)

print(B)
print(B_theta)

