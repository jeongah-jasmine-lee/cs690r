"""
    The code is based on this paper:

    Shavit, Yoli, and Itzik Klein. "Boosting inertial-based human activity 
recognition with transformers." IEEE Access 9 (2021): 53540-53547.

"""

import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from preprocess import *

class CustomDataset(Dataset):
    def __init__(self, filenames, filter=False, sample_rate=60):
        self.filenames = filenames
        self.filter = filter
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = np.load(filename)
        time, torso, left_arm, right_arm = extractMoCap(data)
        time, left_acc, right_acc, left_gyro, right_gyro = extractIMU(data)

        if self.filter:
            # filter_func = fuse_and_rotate
            filter_func = filter_imu
            left_acc_filt, left_gyro_filt = filter_func(left_acc, left_gyro, sample_rate=self.sample_rate)
            right_acc_filt, right_gyro_filt = filter_func(right_acc, right_gyro, sample_rate=self.sample_rate)
            imu = np.stack([left_acc_filt, right_acc_filt, left_gyro_filt, right_gyro_filt], axis=1) # (T, 4, 3)
        else:
            imu = np.stack([left_acc, right_acc, left_gyro, right_gyro], axis=1) # (T, 4, 3)
        mocap = np.concatenate([left_arm, right_arm], axis=1) # (T, 8, 3)

        return {'time': torch.tensor(time, dtype=torch.float32),
                'mocap': torch.tensor(mocap, dtype=torch.float32),
                'imu': torch.tensor(imu, dtype=torch.float32)}

class IMUDataset(Dataset):
    """
    Sliding-window dataset for IMU → MoCap regression.
    
    Args:
        filenames (List[str]): list of .npz files.
        window_size (int): number of time steps per window.
        window_shift (int, optional): step between window starts. 
            Defaults to window_size (non-overlapping).
        filter (bool, optional): if True, apply fuse_and_rotate() to each file.
        sample_rate (float, optional): passed to fuse_and_rotate().
    """
    def __init__(self,
                 filenames,
                 window_size,
                 window_shift=None,
                 filter=False,
                 sample_rate=60.0):
        super().__init__()
        self.window_size = window_size
        self.window_shift = window_shift or window_size
        self.filter = filter
        self.sample_rate = sample_rate
        
        # Load & preprocess each file once
        self._files = []
        for fn in filenames:
            data = np.load(fn)
            # extract raw IMU + MoCap
            _, left_acc, right_acc, left_gyro, right_gyro = extractIMU(data)
            _, _torso, left_arm, right_arm     = extractMoCap(data)
            
            # optionally filter/rotate to global
            if self.filter:
                filter_func = vqf_filter
                left_acc, left_gyro   = filter_func(left_acc, left_gyro, sample_rate=self.sample_rate)
                right_acc, right_gyro = filter_func(right_acc, right_gyro, sample_rate=self.sample_rate)
            
            # stack IMU into shape (T, 4, 3)
            imu = np.stack(
                [left_acc, right_acc, left_gyro, right_gyro],
                axis=1
            )
            # stack MoCap into shape (T, 8, 3)
            mocap = np.concatenate([left_arm, right_arm], axis=1)
            # timestamps
            time = data['time']
            
            self._files.append({
                'time':  time,
                'imu':   imu,
                'mocap': mocap
            })
        
        # build (file_idx, start_idx) index for every window
        self._index = []
        for fid, d in enumerate(self._files):
            n = d['time'].shape[0]
            starts = range(0, n - window_size + 1, self.window_shift)
            for s in starts:
                self._index.append((fid, s))
        
        logging.info(f"IMUDataset: {len(self._files)} files → {len(self._index)} windows")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fid, start = self._index[idx]
        d = self._files[fid]
        end = start + self.window_size
        
        # slice out one window
        time_win  = d['time'][start:end]                      # (window_size,)
        imu_win   = d['imu'][start:end, :, :]                 # (window_size, 4, 3)
        mocap_win = d['mocap'][start:end, :, :]               # (window_size, 8, 3)
        
        return {
            'time':  torch.tensor(time_win,  dtype=torch.float32),
            'imu':   torch.tensor(imu_win,   dtype=torch.float32),
            'mocap': torch.tensor(mocap_win, dtype=torch.float32),
        }