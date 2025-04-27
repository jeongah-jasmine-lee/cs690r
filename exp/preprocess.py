import os, glob
import numpy as np
from scipy.signal import resample, butter, filtfilt
from ahrs.filters import *
import copy
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

GRAVITY_CONSTANT = 9.80665


# ---------- 1.  PREPROCESSING ------------------------------------------------------

# TRIM DATA

# DENOISING
def butter_filter(x, fs, cutoff, btype='low', order=6):
    nyquist = 0.5 * fs
    b, a = butter(order, np.asarray(cutoff) / nyquist, btype=btype)
    return filtfilt(b, a, x, axis=0)

def filter_imu(acc, gyro, fs=60.0, lp_cut=20, bp=(0.1, 20)):
    '''
    Lowpass gyro, lowpass + bandpass acc (for addressing 
    integration drift and high-frequency noise) 
    '''
    gyro_filtered = butter_filter(gyro, fs, lp_cut, 'low', order=6)
    acc_lp = butter_filter(acc, fs, lp_cut, 'low', order=6)
    acc_filtered = butter_filter(acc_lp, fs, bp, 'bandpass', order=2)
    return acc_filtered, gyro_filtered          # shapes (T,3) each

# RESAMPLING

# TRANSFORM DATA'S COORDINATES FROM LOCAL TO GLOBAL (From Assignment 1)
def fuse_and_rotate(local_acc, local_gyr, sample_rate=60.0):
    '''
    Input Parameters
    ----------
    local_data : sensor data object in sensor's coordinates
    
    Output Parameters
    ----------
    global_data : transformed data object in global coordinates
        global_data.sample_rate: the synchornized sampling rate
        global_data.acc: global 3D accelerometer
        global_data.gyr: global 3D gyroscope
    '''

    # Calculate sensor orientation, then use orientation to 
    # transform acceleration and angular velocity from sensor local 
    # frame to global frame
    # Remove gravity from global acceleration
    ahrs_filter = AQUA(acc=local_acc, gyr=local_gyr, frequency=sample_rate)
    
    # Get orientation from AHRS filter
    orientation = ahrs_filter.Q
    
    # Adjust orientation to fit scipy.spatial.transform
    orientation = orientation[:, [1, 2, 3, 0]]
    
    # Transform acceleration from sensor's coordinates to global coordinates
    global_acc = R.from_quat(orientation).apply(local_acc, inverse=True)
    global_gyr = R.from_quat(orientation).apply(local_gyr, inverse=True)
    
    # Adjust for gravity
    global_acc[:, 2] -= GRAVITY_CONSTANT

    # Apply filtering_and_integrate to global data
    global_acc_filt, global_gyr_filt = filter_imu(global_acc, global_gyr, fs=sample_rate, lp_cut=20, bp=(0.1, 20))
    
    return global_acc_filt, global_gyr_filt

# Exercise 2.5.1 Implement the `rodrigues_gravity_removal` function
def rodrigues_gravity_removal(local_acc, local_gyr, sample_rate=60.0):
    '''
    Input Parameters
    ----------
    local_data : sensor data object in sensor's coordinates
    
    Output Parameters
    ----------
    gravity_free_data : compute gravity-free data
        gravity_free_data.sample_rate: the synchornized sampling rate
        gravity_free_data.acc: gravity-free 3D accelerometer
    '''
    
    ####################################################################
    # TODO: Implement rodrigues_gravity_removal
    
    # Global normalized gravity (assuming Z direction)
    G = np.array([0, 0, 1])
    
    # Get coefficients for low-pass filter
    b, a = butter(2, 2 / (sample_rate / 2), btype='lowpass')

    # Get local gravity
    gravL = filtfilt(b, a, local_acc, axis=0)
    gravLnorm = np.linalg.norm(gravL, axis=1)

    # Get rotation axis and angle
    k = np.cross(gravL, G, axisa=1)
    k /= np.linalg.norm(k, axis=1)[:, None]
    theta = np.arccos(np.dot(gravL, G) / gravLnorm)
    theta = theta[:, None]

    # Rotate using Rodrigues and remove gravity
    gravity_free_acc = (local_acc * np.cos(theta) + np.cross(k, local_acc, axisa=1, axisb=1) * np.sin(theta)
            + k * (k * local_acc).sum(1)[:,None] * (1 - np.cos(theta)))
    gravity_free_acc -= G[None,:] * gravLnorm[:,None]
    ####################################################################
    
    # Apply filtering_and_integrate to global data
    acc_filt, gyr_filt = filter_imu(gravity_free_acc, local_gyr, fs=sample_rate, lp_cut=20, bp=(0.1, 20))

    return acc_filt, gyr_filt