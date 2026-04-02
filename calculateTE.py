import numpy as np

# JIDT requirements
import jpype
from jpype import *
from jpype import JArray, JDouble
import os
import sys

## Start JVM functionality ##
jarLocation = 'jidt/infodynamics.jar'
if not os.path.isfile(jarLocation):
    sys.exit("infodynamics.jar not found (expected at " + os.path.abspath(jarLocation) + ")")

if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        "--enable-native-access=ALL-UNNAMED",
        "-ea",
        f"-Djava.class.path={jarLocation}"
    )

def TE_KSG_linvel(vel, i, j, k):
    """
    Calculates the local Transfer Entropy from the velocity of two particles
    Using the JIDT toolkit with the Kraskov-Gasser (KSG) estimation method.
    Multivariate Kraskov

    Args:
        vel (np.ndarray): Velocity data with shape (T, N, 2)
                          (time, particle, [x, y]).
        i (int): Index of the destination particle.
        j (int): Index of the source particle.
        k (int): Embedding length (history length) for the TE calculation.

    Returns:
        np.ndarray: A 1-D numpy array of local TE values for each time step.
    """
    # Initialize/set-up the calculator
    teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorMultiVariateKraskov
    teCalc = teCalcClass()
    
    # Set properties for the calculator
    teCalc.setProperty("k", "4")  # Number of nearest neighbors for KSG estimation (typically rule is 4 for small datasets; <200 points)
    teCalc.setProperty("NORMALISE", "true")  # Normalize the data
    teCalc.setProperty("NOISE_LEVEL_TO_ADD", "0")  # No noise addition
    
    # Extract velocity data for destination (i) and source (j) particles
    # Shape: (T, 2) for each particle
    dest_data = vel[:, i, :]  # Destination particle velocity
    source_data = vel[:, j, :]  # Source particle velocity
    
    # Get dimensions
    T = dest_data.shape[0]  # Number of time points
    source_dim = source_data.shape[1]  # Should be 2 (x, y components)
    dest_dim = dest_data.shape[1]    # Should be 2 (x, y components)
    
    # Initialize the calculator with actual data dimensions
    # Parameters: (sourceDimensions, destDimensions, k, k_tau, l, l_tau, delay)
    l = k
    teCalc.initialise(source_dim, dest_dim, k, 1, l, 1, 1)

    # Convert numpy arrays to Java arrays
    # JIDT multivariate expects [time][variables] format
    dest_java = jpype.JArray(jpype.JDouble, 2)([dest_data[t, :].tolist() for t in range(T)])
    source_java = jpype.JArray(jpype.JDouble, 2)([source_data[t, :].tolist() for t in range(T)])
    
    # Set observations - source comes first, then destination
    teCalc.setObservations(source_java, dest_java)
    
    # Compute local transfer entropy values
    local_te_java = teCalc.computeLocalOfPreviousObservations()
    
    # Convert Java array back to numpy array
    local_te = np.array([local_te_java[t] for t in range(len(local_te_java))])

    return local_te

def TE_KSG_angvel(vel, i, j, k):
    """
    Calculates the local Transfer Entropy from the angular velocity of two particles:
        source value: heading_j - heading_i
        target value: heading_j - heading_j
    Using the JIDT toolkit with the Kraskov-Gasser (KSG) estimation method.
    Univariate Continuous Kraskov

    Args:
        vel (np.ndarray): Velocity data with shape (T, N, 2)
                          (time, particle, [x, y]).
        i (int): Index of the destination particle.
        j (int): Index of the source particle.
        k (int): Embedding length (history length) for the TE calculation.

    Returns:
        np.ndarray: A 1-D numpy array of local TE values for each time step.
    """
    # Initialize/set-up the calculator
    teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
   
    # Set properties for the calculator
    teCalc.setProperty("k", "4")  # Number of nearest neighbors for KSG estimation
    teCalc.setProperty("NORMALISE", "true")  # Normalize the data
    teCalc.setProperty("NOISE_LEVEL_TO_ADD", "0")  # No noise addition

    # Initialize the calculator with actual data dimensions
    # Parameters: (k, k_tau, l, l_tau, delay)
    l = k
    teCalc.initialise(k, 1, l, 1, 1)

    # Calculate state variables
    def wrap_angle(angles):
        return (angles + np.pi) % (2 * np.pi) - np.pi

    headings = np.unwrap(np.arctan2(vel[:, :, 1], vel[:, :, 0]), axis=0)  # (T, N)
    angular_velocity = np.diff(headings, axis=0)  # (T-1, N)

    omega_i = angular_velocity[:, i]
    omega_j = angular_velocity[:, j]

    source_value = wrap_angle(omega_j - omega_i)[:-1]
    
    target_value = omega_i[:-1]

    # Compute TE
    teCalc.setObservations(JArray(JDouble, 1)(source_value), JArray(JDouble, 1)(target_value))
    local_te_java = teCalc.computeLocalOfPreviousObservations()
    local_te = np.array([local_te_java[t] for t in range(len(local_te_java))])
   
    return np.insert(local_te, 0, [0, 0])