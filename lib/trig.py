import math
import numpy as np
import lib.data_formatter as datform


def distance(A : np.ndarray(shape=(2,1)), B : np.ndarray(shape=(2,1))) -> float:
    """Takes two ordered pairs and returns the distance between them"""
    C = np.subtract(A, B)
    r = np.linalg.norm(C)
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
    theta = np.arctan2(A[1, 0], A[0, 0])
    return theta



def rotate(A, theta : float) -> np.ndarray(shape=(2,1)):
    """Rotates a 2d vector about the origin counter-clockwise"""
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], #cos(theta) , -sin(theta)
                        [np.sin(theta), np.cos(theta)]]) #sin(theta) ,  cos(theta)
    A_prime = np.dot(rot_mat, A)
    return A_prime



def flip_and_align(positions : np.ndarray) -> np.ndarray:
    """Flips the y-coordinates of the positions to align with the positive x-axis."""
    flipped_positions = positions.copy()
    flipped_positions[1, :] = -flipped_positions[1, :]  # invert the y-coordinates
    flipped_positions[1, :] -= flipped_positions[1,-1]  # translate to have base align with positive x 
    return flipped_positions



def normalise_minor(A : np.ndarray(shape=(2, 5)), Base : np.ndarray(shape=(2,1))) -> np.ndarray(shape=(2, 5)):
    """takes an array of positions and rotates them such that an inputted unit vector (base) would be parallel to the positive x-axis"""
    theta = theta_from_posx(Base)
    theta = 2*math.pi - theta
    A_prime = rotate(A, theta)
    return A_prime

def normalise_major(A):
    """takes a 3d array of positions of tracking dots and normalises them by aligning the G-YN line with the positive x-axis"""
    shape = A.shape
    L = shape[2]

    in_set = datform.a_3d_slice(A, axis=2)
    out_set = []

    for n in range(L):
        current = in_set[n]
        U = current[:, -2:-1]
        V = current[:, -1:]
        b_hat = origin_unit(U, V)
        rotated = normalise_minor(current, b_hat)
        aligned = flip_and_align(rotated)
        out_set.append(np.array(aligned))

    out_array = datform.a_3d_recombine(out_set, axis=2)
    return out_array

def pythag_inf(A) -> float:
    """takes a list of floats and returns the pythag of all of them"""
    internal = 0
    for i in A:
        internal += float(i ** 2)
    return math.sqrt(internal)