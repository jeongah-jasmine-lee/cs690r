import os, glob
import numpy as np
from scipy.signal import butter, filtfilt
from ahrs.filters import *
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import DataLoader, Dataset
from vqf import VQF

GRAVITY_CONSTANT = 9.80665

# ---------- LOAD ------------------------------------------------------
def extractMoCap(data):
    time = data['time'] # (T,)
    torso = np.stack([np.zeros_like(data['leftShoulderPosRel']),
                      data['leftShoulderPosRel'],
                      data['rightShoulderPosRel'],
                      np.zeros_like(data['rightShoulderPosRel'])], axis=1) # (T, 4, 3)
    left_arm = np.stack([data['leftShoulderPosRel'],
                         data['leftElbowPosRel'],
                         data['leftWristPosRel'],
                         data['leftFingerPosRel']], axis=1) # (T, 4, 3)
    right_arm = np.stack([data['rightShoulderPosRel'],
                          data['rightElbowPosRel'],
                          data['rightWristPosRel'],
                          data['rightFingerPosRel']], axis=1) # (T, 4, 3)

    return time, torso, left_arm, right_arm

def extractIMU(data):
    time = data['time'] # (T,)
    left_acc = data['accelerationLeftLoc'] # (T, 3)
    right_acc = data['accelerationRightLoc'] # (T, 3)
    left_gyro = data['gyroLeftLoc'] # (T, 3)
    right_gyro = data['gyroRightLoc'] # (T, 3)

    return time, left_acc, right_acc, left_gyro, right_gyro

# ---------- PREPROCESSING ------------------------------------------------------

# DENOISING
def butter_filter(x, sample_rate, cutoff, btype='low', order=6):
    nyquist = 0.5 * sample_rate
    b, a = butter(order, np.asarray(cutoff) / nyquist, btype=btype)
    return filtfilt(b, a, x, axis=0)

def filter_imu(acc, gyro, sample_rate=60.0, lp_cut=20, bp=(0.2, 20)):
    '''
    Lowpass gyro, lowpass + bandpass acc (for addressing 
    integration drift and high-frequency noise) 
    '''
    gyro_filtered = butter_filter(gyro, sample_rate, lp_cut, 'low', order=6)
    acc_lp = butter_filter(acc, sample_rate, lp_cut, 'low', order=6)
    acc_filtered = butter_filter(acc_lp, sample_rate, bp, 'bandpass', order=2)
    return acc_filtered, gyro_filtered          # shapes (T,3) each

def hp_filter_imu(acc, gyro, sample_rate=60.0, hp_cut=0.1):
    acc_filtered = butter_filter(acc, sample_rate, cutoff=hp_cut,btype='high', order=6)
    return acc_filtered, gyro          # shapes (T,3) each

# NORMALIZATION
def imu_normalization_3D(x):
    signal_mean = x.mean(axis=0)
    signal_std = x.std(axis=0)
    return (x - signal_mean) / (signal_std + 1e-6)

# TRANSFORM DATA COORDINATES FROM LOCAL TO GLOBAL (From Assignment 1)
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
    global_acc, global_gyr = filter_imu(global_acc, global_gyr, sample_rate=sample_rate, lp_cut=20, bp=(0.1, 20))
    
    return global_acc, global_gyr

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
    
    # Apply filtering_and_integrate to global data
    acc_filt, gyr_filt = filter_imu(gravity_free_acc, local_gyr, fs=sample_rate, lp_cut=10, bp=(0.2, 10))

    return acc_filt, gyr_filt

"""
    *** vpg_filter IS UNDER ACTIVE DEVELOPMENT ***
"""
def vqf_filter(local_acc: np.ndarray,
               local_gyr: np.ndarray,
               sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Valo Quaternion Filter (VQF) to estimate orientation,
    rotate IMU data to global frame, and remove gravity.

    Args:
        local_acc: (T,3) accelerometer in sensor frame [m/s²]
        local_gyr: (T,3) gyroscope in sensor frame [rad/s]
        sample_rate: sampling frequency [Hz]

    Returns:
        global_acc: (T,3) accelerometer in global frame, gravity removed
        global_gyr: (T,3) gyroscope in global frame
    """
    # 1) Run VQF to get orientation quaternions (w,x,y,z) shape (T,4)
    vqf = VQF(sample_rate)
    out = vqf.updateBatch(local_gyr, local_acc)
    quats = out['quat6D']  # → np.ndarray (T,4)

    # 2) Convert to scipy Rotation (expects [x,y,z,w])
    #    so reorder quats from (w,x,y,z) → (x,y,z,w)
    rot = R.from_quat(quats[:, [1, 2, 3, 0]])

    # 3) Rotate into global frame
    global_acc = rot.apply(local_acc, inverse=True)
    global_gyr = rot.apply(local_gyr, inverse=True)

    # 4) Remove gravity on global Z
    global_acc[:, 2] -= GRAVITY_CONSTANT

    return global_acc, global_gyr