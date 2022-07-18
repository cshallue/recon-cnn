import numpy as np


def propagator(cross_spectrum, target_power_spectrum):
    return cross_spectrum / target_power_spectrum


def correlation_coefficient(cross_spectrum, predicted_power_spectrum,
                            target_power_spectrum):
    return cross_spectrum / np.sqrt(
        predicted_power_spectrum * target_power_spectrum)


def transfer_function(predicted_power_spectrum, target_power_spectrum):
    return np.sqrt(predicted_power_spectrum / target_power_spectrum)
