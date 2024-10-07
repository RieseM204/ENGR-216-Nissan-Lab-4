"""Library for working with forces in order to solve classical mechanics problems"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import math

def calc_fric_k(theta, a):                                  # Take the angle and measure acceleration as inputs
    g = 9.81                                                # Acceleration due to gravity in m/s^2
    coeff = math.tan(theta) - (a)/(g * math.cos(theta))     # Calculate the coefficient using my derrived equation
    return coeff                                            # Return the coefficient

def calc_fric_s(theta):                                     # Take the minimum angle at which the block accelerates as input
    return math.tan(theta)                                  # Return the tangent of the angle (it really is that simple)