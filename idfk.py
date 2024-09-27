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

mat = str(input("material :"))
f_t = str(input("friction type: "))
n = int(input("n: "))

df = pd.read_csv(f"bin/friction_{mat}_{f_t}_{n}.csv")
df.dropna(axis="index", how="any")

print(df)