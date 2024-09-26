import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mat_list = ['rubber', 'wood']
type_list = ['static', 'kinetic']
color_list = ['darkorange', 'lightorange', 'hotpink', 'yellowneon', 'green']
plot_color_list = ['darkorange', 'orange', 'pink', 'yellow', 'green']

def plotter(A, B):
    for n in range(1,5):
        df = pd.read_csv(f'Data/friction_{A}_{B}_{n}.csv')
        for i in range(5):
            pos_x = df[f'position_px_x-{color_list[i]}'][3:]
            pos_y = df[f'position_px_y-{color_list[i]}'][3:]
            plt.plot(pos_x, pos_y, label=i, color=plot_color_list[i])
        plt.title(f"{A} {B} {n}")
        plt.show()

for A in mat_list:
    for B in type_list:
        plotter(A, B)

