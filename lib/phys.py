"""Library for working with forces in order to solve classical mechanics problems"""
# WIP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import math

def calc_fric_k(theta, a):
    g = 9.81
    coeff = math.cot(theta) - (a)/(g * math.sin(theta))
    return coeff

def calc_fric_s(theta):
    return math.tan(theta)