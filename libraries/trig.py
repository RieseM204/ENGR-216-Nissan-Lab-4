import math
import numpy as np

def distance(A : np.ndarray(shape=(1,2)), B : np.ndarray(shape=(1,2))) -> float:
    "Takes two ordered pairs and returns the distance between them"
    A = A.tolist()
    B = B.tolist()
    dx = float(B[0])-float(A[0])
    dy = float(B[1])-float(A[1])
    r = math.sqrt((dx**2) + (dy**2))
    return r

def origin_roll(A : np.ndarray(shape=(1,2)), B : np.ndarray(shape=(1,2))) -> np.ndarray(shape=(1,2)):
    "Takes two 1x2 arrays and returns a vector from the origin assuming that A is closest to the origin"
    V = [float(B[0]-A[0]), float(B[1]-A[1])]
    V_out = np.array(V)
    return V_out

def unit(v : np.ndarray(shape=(1, 2))) -> np.ndarray(shape=(1, 2)):
    "Takes a 1x2 vector and converts it into a unit vector"
    l = np.linalg.norm(v)
    v_hat = v / l
    return v_hat

def origin_unit(A : np.ndarray(shape=(1,2)), B : np.ndarray(shape=(1,2))) -> np.ndarray(shape=(1, 2)):
    "Takes two 1x2 arrays and returns a unit vector from the origin assuming that A is closest to the origin"
    v = origin_roll(A, B)
    v_hat = unit(v)
    return v_hat

def theta_between(A : np.ndarray(shape=(1,2)), B : np.ndarray(shape=(1,2))) -> float:
    "Takes two 1x2 unit vectors and returns the angle between them in radians"
    AB_mag_prod = np.linalg.norm(A) * np.linalg.norm(B)
    AB_dot = np.dot(A, B)
    theta = math.acos(AB_dot / AB_mag_prod)
    return theta

def theta_from_posx(A : np.ndarray(shape=(1,2))) -> float:
    "Takes a 1x2 unit vector and returns the internal angle in radians from the positive x-axis"
    B = np.array([1, 0])
    theta = theta_between(B, A)
    return theta
