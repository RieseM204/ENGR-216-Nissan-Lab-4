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

rs3a = datform.trial_to_3d('rubber', 'static', 3)

print(rs3a)

rs3a_norm = trig.normalise_major(rs3a)

print(rs3a_norm)

def set_plot(set, title):
    set_split = datform.a_3d_slice(set, axis=0)
    for i in range(len(set_split)):
        c = plot_color_list[i]
        current = set_split[i]
        x = current[0].tolist()
        y = current[1].tolist()
        plt.plot(x, y, color=c)
    plt.title(title)
    plt.show()

set_plot(rs3a, "rubber static 3")
set_plot(rs3a_norm, "rubber static 3 normalized")