import os                                                                           # Import OS to access directories
import numpy as np                                                                  # Import NumPy for just about everything
import pandas as pd                                                                 # Import Pandas to work with the data
import math                                                                         # Import math just in case
from scipy.signal import savgol_filter                                              # Import savgol filter just in case
import lib.data_formatter as datform                                                # Import my data formatting library
import lib.trig as trig                                                             # Import my trigonometry library
import lib.phys as phys                                                             # Import my physics equation library
import matplotlib.pyplot as plt

"""Main file, intended to be run"""

#stack order: DO, LO, HP, G, YN

#units are m, kg, s

mass = 0.2241           # kg

LO_HP = 0.178           # m
G_YN = 0.217            # m

A_r = 0.00378           # m^2
A_r_std = 0.00000071
A_w = 0.00364           # m^2
A_w_std = 0.00000071

def px_to_m(A) -> float:
    """Takes in the data for a timestep and returns the number to multiply px distance by to get m distance"""
    LO = A[:, 1].reshape(2, 1)                                                      # Get the positional data for Light Orange
    HP = A[:, 2].reshape(2, 1)                                                      # Get the positional data for Hot Pink
    G = A[:, 3].reshape(2, 1)                                                       # Get the positional data for Green
    YN = A[:, 4].reshape(2, 1)                                                      # Get the positional data for Yellow Neon
    distLOHP = trig.distance(LO, HP)                                                # Get the Light Orange - Hot Pink distance
    distGYN = trig.distance(G, YN)                                                  # Get the Green - Yellow Neon distance
    factorLOHP = LO_HP / distLOHP                                                   # Divide the measured length (in meters) of LO-HP by the pixel distance
    factorGYN = G_YN / distGYN                                                      # Divide the measured length (in meters) of G-YN by the pixel distance
    factor = (factorLOHP + factorGYN)/2                                             # Take the average
    return factor                                                                   # Return that as a conversion factor

def static(material : str):
    """Does all the static friction stuff"""
    test_list = [i for i in os.listdir("bin/") if f"{material}_static" in i]        # Getting a list of all of the static trials for the given material
    length = len(test_list)                                                         # Getting the length of that list
    full_coeff_list = []                                                            # Setting up a list for all of the calculated coefficients
    full_ang_list = []
    for i in range(1, length + 1):                                                  # Looping through all of the trials for that material and friction type

        full = datform.trial_to_3d(material, "static", i, True)                     # Converting trial data to a 3d array
        full_norm = trig.normalise_major(A = full)                                  # Normalising the data to the G and YN (base) dots to make things easier
        full_norm_sliced = datform.a_3d_slice(A = full_norm, axis = 2)              # Slicing up the array into the individual timesteps
        
        sample_norm = np.array(datform.a_3d_slice(A = full_norm, axis = 0)[0:3])    # Taking only the first three dots from the full array

        factor_list = []                                                            # Setting up list for conversion factors
        for j in full_norm_sliced:                                                  # Iterating through the slices of the normalised full data
            factor_list.append(px_to_m(j))                                          # Getting the conversion factors
        factor_list = factor_list[1:-1]                                             # Slicing conversion factor list to line up with later velocity list

        df = pd.read_csv(f"bin/friction_{material}_static_{i}.csv")                 # Pull the data again as a dataframe to get the timestamps
        df = df.dropna(axis='index', how='any')                                     # Drop nans
        timestamp = [i/1000 for i in df['timestamp'].tolist()]                      # Convert to seconds

        DO_n = sample_norm[0]                                                       # Get the Dark Orange data from the sample set
        O_n = sample_norm[1]                                                        # Get the Light Orange data from the sample set
        HP_n = sample_norm[2]                                                       # Get the Hot Pink data from the sample set

        O_HP_unit = np.array([[],[]])                                               # Set up for the Light Orange - Hot Pink unit vector

        for j in range(len(O_n[0])):                                                # Iterate through the data
            dx = HP_n[0][j]-O_n[0][j]                                               # Get the change in x
            dy = O_n[1][j]-HP_n[1][j]                                               # Get the change in y (but inverted)
            O_HP = np.array([[dx], [dy]])                                           # Use that as a vector
            l = np.linalg.norm(O_HP)                                                # Get the magnitude
            O_HP_hat = O_HP / l                                                     # Convert it into a unit vector
            O_HP_unit = np.concatenate((O_HP_unit, O_HP_hat), axis=1)               # Add that to the array of unit vectors

        theta_list = []                                                             # Set up a list for angles (radians)
        shape = O_HP_unit.shape                                                     # Get the shape of the unit vector array
        l = shape[1]                                                                # Find the length of it?
        units = []                                                                  # Set up a list of fixed unit vectors

        for j in range(l):                                                          # Iterate through the length of the unit vector array
            current_slice = O_HP_unit[:,j]                                          # Take a slice (current unit vector)
            current_slice = np.reshape(current_slice, (2, 1))                       # Reshape it to the standard 2x1 shape
            units.append(np.array(current_slice))                                   # Add that to the list of fixed unit vectors

        for j in units:                                                             # Iterate through the fixed unit vectors
            current_theta = trig.theta_from_posx(j)                                 # Find the angle from the positive x axis
            theta_list.append(float(current_theta))                                 # Append to the list of angles

        dx_list = []                                                                # Set up a list of horizontal displacements

        for j in range(len(DO_n[0])):                                               # Iterate for the length of the Dark Orange data
            dx = DO_n[0][j]-O_n[0][j]                                               # Find the horizontal displacement between the dark orange and light orange dots (even though it isn't properly normalised. I've accepted this as space wizard magic)
            dx_list.append(float(dx))                                               # Add that to the horizontal displacement list
            
        ts_list, v_list = datform.fdiff(x = timestamp, y = dx_list, use_smoothing = True)   # Use finite differences to get the velocities and new timestamps
        v_list_new = []                                                             # Set up a list for converted velocities
        for j in range(len(v_list)):                                                # Iterate through the length of the velocity list
            factor = factor_list[j]                                                 # Get the corresponding conversion factor
            v_list_new.append(float(v_list[j]*factor))                              # Multiply and append to the converted velocity list

        theta_list = theta_list[2:-2]                                               # Correct the list of angles to be the right length

        tsa_list, a_list = datform.fdiff(x = ts_list, y = v_list_new, use_smoothing = True)  # Use finite differences to get the accelerations

        a_vs_ang = np.array([[list(a_list)],[list(theta_list)]])                    # Put the accelerations and angles together into an array

        thresh = 0.5                                                                # Set a threshold to eliminate low-amplitude noise

        useful_ang_list = a_vs_ang[1, a_vs_ang[0] > thresh]                         # Use Boolean indexing to return only the angle with accelerations above the threshold
        useful_ang = min(useful_ang_list)                                           # Get the minimum angle at which the threshold is exceeded
        full_ang_list.append(float(useful_ang))
        coeff = phys.calc_fric_s(useful_ang)                                        # Use that angle to get the coefficient
        full_coeff_list.append(float(coeff))                                        # Add that coefficient to the full list
        plt.plot(theta_list, a_list, label=i)
    std = np.std(full_coeff_list)                                                   # Get the standard deviation of the coefficients
    mean_coeff = np.mean(full_coeff_list)                                           # Get the mean of the coefficients
    mean_u_ang = np.mean(full_ang_list)
    std_u_ang = np.std(full_ang_list)
    plt.title(f"{material} static: acceleration vs angle")
    plt.xlabel("Angle (Radians)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.axvline(x = mean_u_ang, color = "black")
    plt.axvline(x = mean_u_ang - std_u_ang, color = "black", linestyle = "--")
    plt.axvline(x = mean_u_ang + std_u_ang, color = "black", linestyle = "--")
    plt.axvspan(mean_u_ang - std_u_ang, mean_u_ang + std_u_ang, color='yellow', alpha=0.2)
    plt.legend()
    plt.show()
    return(mean_coeff, std)                                                         # Return the mean and the standard deviation
        

def kinetic(material : str):
    """Does all the kinetic friction stuff"""
    test_list = [i for i in os.listdir("bin/") if f"{material}_kinetic" in i]       # Everything here is the same as the static function until line 182
    length = len(test_list)
    super_coeff_list = []
    for i in range(1, length + 1):

        full = datform.trial_to_3d(material, "kinetic", i, True)
        full_norm = trig.normalise_major(A = full)
        full_norm_sliced = datform.a_3d_slice(A = full_norm, axis = 2)
        
        sample = np.array(datform.a_3d_slice(A = full_norm, axis = 0)[0:3])
        sample_sliced = datform.a_3d_slice(A = sample, axis = 2)

        factor_list = []
        for j in full_norm_sliced:
            factor_list.append(px_to_m(j))
        factor_list = factor_list[1:-1]

        df = pd.read_csv(f"bin/friction_{material}_kinetic_{i}.csv")
        df = df.dropna(axis='index', how='any')
        timestamp = [i/1000 for i in df['timestamp'].tolist()]

        DO_n = sample[0]
        O_n = sample[1]
        HP_n = sample[2]

        O_HP_unit = np.array([[],[]])

        for j in range(len(O_n[0])):
            dx = HP_n[0][j]-O_n[0][j]
            dy = O_n[1][j]-HP_n[1][j]
            O_HP = np.array([[dx], [dy]])
            l = np.linalg.norm(O_HP)
            O_HP_hat = O_HP / l
            O_HP_unit = np.concatenate((O_HP_unit, O_HP_hat), axis=1)

        theta_list = []
        shape = O_HP_unit.shape
        l = shape[1]
        units = []

        for j in range(l):
            current_slice = O_HP_unit[:,j]
            current_slice = np.reshape(current_slice, (2, 1))
            units.append(np.array(current_slice))

        for j in units:
            current_theta = trig.theta_from_posx(j)
            theta_list.append(float(current_theta))

        dx_list = []

        for j in range(len(DO_n[0])):
            dx = DO_n[0][j]-O_n[0][j]
            dx_list.append(float(dx))
            
        ts_list, v_list = datform.fdiff(x = timestamp, y = dx_list, use_smoothing = True)
        v_list_new = []
        for j in range(len(v_list)):
            factor = factor_list[j]
            v_list_new.append(float(v_list[j]*factor))

        tsa_list, a_list = datform.fdiff(x = ts_list, y = v_list_new, use_smoothing = True)
        theta_list = theta_list[2:-2]
        coeff_list = []                                                             # Set up a list of coefficients
        for j in range(len(a_list)):                                                # Iterate for the length of the acceleration list
            coeff_list.append(float(phys.calc_fric_k(theta_list[j], a_list[j])))    # Calculate the kinetic friction coefficient for each acceleration value and corresponding angle, and append to the coefficient list
        
        super_coeff_list.extend(list(coeff_list))
        coeff = np.mean(coeff_list)                                                 # Find the mean of the coefficients
        coeff_std = np.std(coeff_list)                                              # Find the standard deviation of the coefficients

    std = np.std(super_coeff_list)                                                # Essentially apply a huge pythag theorem to all stds to get the propogated std
    mean_coeff = np.mean(super_coeff_list)                                           # Get the mean coefficient out of all of the coefficients
    plt.hist(super_coeff_list, bins = int(math.sqrt(len(super_coeff_list))))
    plt.title(f"{material} kinetic: friction coefficients")
    plt.ylabel("n")
    plt.xlabel("coefficient of kinetic friction")
    plt.axvline(x = mean_coeff, color = "black")
    plt.axvline(x = mean_coeff - std, color = "black", linestyle = "--")
    plt.axvline(x = mean_coeff + std, color = "black", linestyle = "--")
    plt.axvspan(mean_coeff - std, mean_coeff + std, color='yellow', alpha=0.2)
    plt.show()
    return(mean_coeff, std)                                                         # Return the mean coefficient and the standard deviation

def main():
    """Everything all together"""
    r_k, r_k_s = kinetic("rubber")                                                  # Calculate the kinetic friction for rubber
    w_k, w_k_s = kinetic("wood")                                                    # Calculate the kinetic friction for wood
    r_s, r_s_s = static("rubber")                                                   # Calculate the static friction for rubber
    w_s, w_s_s = static("wood")                                                     # Calculate the static friction for wood

    r_k_p = r_k_s / r_k * 100
    w_k_p = w_k_s / w_k * 100
    r_s_p = r_s_s / r_s * 100
    w_s_p = w_s_s / w_s * 100

    print(f"rubber kinetic: {r_k:.5f} ± {r_k_s:.5f} or {r_k_p:.2f}%")                                      # Print the kinetic coefficient and std for rubber
    print(f"rubber kinetic by area (m^2): {r_k / A_r:.5f}")
    print(f"wood kinetic: {w_k:.5f} ± {w_k_s:.5f} or {w_k_p:.2f}%")                                        # Print the kinetic coefficient and std for wood
    print(f"wood kinetic by area (m^2): {w_k / A_w:.5f}")
    print(f"rubber static: {r_s:.5f} ± {r_s_s:.5f} or {r_s_p:.2f}%")                                       # Print the static coefficient and std for rubber
    print(f"rubber static by area (m^2): {r_s / A_r:.5f}")
    print(f"wood static: {w_s:.5f} ± {w_s_s:.5f} or {w_s_p:.2f}%")                                         # Print the static coefficient and std for wood
    print(f"wood static by area (m^2): {w_s / A_w:.5f}")

if __name__ == "__main__":                                                          # Generic conditional to declare this as the functional script
    main()                                                                          # Do it all