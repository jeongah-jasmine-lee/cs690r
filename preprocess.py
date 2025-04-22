import os, glob
import numpy as np
from scipy import signal
import torch
from torch.utils.data import Dataset

# ---------- 1.  FILTERS ------------------------------------------------------

def butter_filter(x, fs, cutoff, btype='low', order=6):
    nyquist = 0.5 * fs
    b, a = signal.butter(order, np.asarray(cutoff) / nyquist, btype=btype)
    return signal.filtfilt(b, a, x, axis=0)

def clean_imu(acc, gyro, fs=50, lp_cut=20, bp=(0.1, 20)):
    """Lowpass gyro, lowpass + bandpass acc (for gravity/DC removal) """
    gyro_filtered = butter_filter(gyro, fs, lp_cut, 'low', order=6)
    acc_lp = butter_filter(acc, fs, lp_cut, 'low', order=6)
    acc_filtered = butter_filter(acc_lp, fs, bp, 'bandpass', order=2)
    return acc_filtered, gyro_filtered          # shapes (T,3) each

# ---------- 2.  DATASET & WINDOWING -----------------------------------------

def sliding_windows(signal, win_len=100, step=25):
    """
    Yield fixed length, possibly overlapping windows from signal
    """
    N = signal.shape[0]
    t = 0

    while t < N:
        if t + win_len > N:               # tail case
            start_idx = max(0, N - win_len)
            end_idx   = N
        else:                             # regular case
            start_idx = t
            end_idx   = t + win_len

        # Yield a view (no copy)
        yield signal[start_idx:end_idx]

        # If we just emitted the tail‑aligned window, exit
        if t + win_len > N:
            break

        t += step

class IMUPoseDataset(Dataset):
    """
    Returns (imu_window, pose_window, subj_id)
      • imu_window  : (T, 12)  [accL, gyroL, accR, gyroR]
      • pose_window : (T,  J, 3)  where J = len(joint_keys)
    """
    def __init__(self,
                 root_dir,
                 win_len=100,
                 step=25,
                 fs=50,
                 joint_keys=None):  
        if joint_keys is None:   
            joint_keys = [
                'leftShoulderPosRel',  'leftElbowPosRel',
                'leftWristPosRel',     'leftFingerPosRel',
                'rightShoulderPosRel', 'rightElbowPosRel',
                'rightWristPosRel',    'rightFingerPosRel',
            ]

        self.samples, self.subj_ids = [], []
        acc_all, gyro_all, raw = [], [], []

        # Load and Filter
        for f in sorted(glob.glob(os.path.join(root_dir, '*.npz'))):
            data = np.load(f)

            # IMU
            accL, gyroL = data['accelerationLeftLoc'],  data['gyroLeftLoc']
            accR, gyroR = data['accelerationRightLoc'], data['gyroRightLoc']
            accL, gyroL = clean_imu(accL, gyroL, fs)
            accR, gyroR = clean_imu(accR, gyroR, fs)

            # MoCap : stack selected joints -> (T, J, 3)
            pose = np.stack([data[k] for k in joint_keys], axis=1)

            raw.append((accL, gyroL, accR, gyroR, pose, _subject_id(f)))
            acc_all.extend([accL, accR]);  gyro_all.extend([gyroL, gyroR])

        # Normalization
        self.acc_mu  = np.mean(np.vstack(acc_all),  axis=0)
        self.acc_std = np.std( np.vstack(acc_all),  axis=0) + 1e-6
        self.gyro_mu = np.mean(np.vstack(gyro_all), axis=0)
        self.gyro_std= np.std( np.vstack(gyro_all), axis=0) + 1e-6

        # Sliding Windows
        for accL, gyroL, accR, gyroR, pose, sid in raw:
            accL  = (accL - self.acc_mu)  / self.acc_std
            accR  = (accR - self.acc_mu)  / self.acc_std
            gyroL = (gyroL - self.gyro_mu) / self.gyro_std
            gyroR = (gyroR - self.gyro_mu) / self.gyro_std

            imu = np.concatenate([accL, gyroL, accR, gyroR], axis=1)  # (T, 12)

            for imu_w, pose_w in zip(sliding_windows(imu,  win_len, step),
                                     sliding_windows(pose, win_len, step)):
                self.samples.append((imu_w.astype('float32'),
                                     pose_w.astype('float32')))
                self.subj_ids.append(sid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imu, pose = self.samples[idx]
        return torch.from_numpy(imu), torch.from_numpy(pose), self.subj_ids[idx]

# ---------- Helpers ----------------------------------------------------------
def _subject_id(path):
    base = os.path.basename(path)
    return int(base.split('_')[-2][1:])   # 'sXX' -> XX