import math
import numpy as np



def distance(A : np.ndarray(shape=(2,1)), B : np.ndarray(shape=(2,1))) -> float:
    """Takes two ordered pairs and returns the distance between them"""
    C = np.subtract(A, B)
    C_dot = np.dot(C, C)
    r = math.sqrt(C_dot)
    return r



def unit(v : np.ndarray(shape=(2,1))) -> np.ndarray(shape=(2,1)): 
    """Takes a 2x1 vector and converts it into a unit vector"""
    l = np.linalg.norm(v)
    v_hat = v / l
    return v_hat



def origin_unit(A : np.ndarray(shape=(2,1)), B : np.ndarray(shape=(2,1))) -> np.ndarray(shape=(2,1)):
    """Takes two 2x1 arrays and returns a unit vector from the origin assuming that A is closest to the origin"""
    v = np.subtract(B, A)
    v_hat = unit(v)
    return v_hat



def theta_between(A : np.ndarray(shape=(2,1)), B : np.ndarray(shape=(2,1))) -> float:
    """Takes two 2x1 unit vectors and returns the angle between them in radians"""
    AB_mag_prod = np.linalg.norm(A) * np.linalg.norm(B)
    AB_dot = np.dot(A, B)
    theta = math.acos(AB_dot / AB_mag_prod)
    return theta



def theta_from_posx(A : np.ndarray(shape=(2,1))) -> float:
    """Takes a 2x1 unit vector and returns the internal angle in radians from the positive x-axis"""
    B = np.array([1, 0])
    theta = theta_between(A, B)
    return theta



def rotate_cc(A : np.ndarray(shape=(2,1)), theta : float) -> np.ndarray(shape=(2,1)):
    """Rotates a 2x1 vector about the origin counter-clockwise"""
    rot_mat = np.array([[float(math.cos(theta)), float(math.sin(theta) * -1)], #cos(theta) , -sin(theta)
                        [float(math.sin(theta)), float(math.cos(theta))]])     #sin(theta) ,  cos(theta)
    A_prime = np.multiply(rot_mat, A)
    return A_prime



def rotate_cw(A : np.ndarray(shape=(2,1)), theta : float) -> np.ndarray(shape=(2,1)):
    """Rotates a 2x1 vector about the origin clockwise"""
    t = math.pi * 2
    theta_new = t - theta
    A_prime = rotate_cc(A, theta_new)
    return A_prime



def normalise(Set : np.ndarray(shape=(2, 5)), Base : np.ndarray(shape=(2, 1)), Axis : int) -> np.ndarray(2, 5):
    """fuck"""
    # I have to make this but it's going to be annoying
    raise NotImplementedError