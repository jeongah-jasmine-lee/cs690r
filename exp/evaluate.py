import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ConvTransformer, SimpleNN, BiLSTM
from losses import angle_constraint_loss, bone_length_consistency_loss
from dataset import *


filenames = glob.glob('../data/*.npz')
assert len(filenames) > 0, 'No data files found in ./data/'

filename = filenames[1]
data = np.load(filename)
print('Data containing:')
[print(f'Array {f} of shape {data[f].shape}') for f in data.files];

dt = np.diff(data['time'])
SAMPLE_RATE = 1.0 / np.mean(dt)
print('Sample rate:', SAMPLE_RATE)


subjects = np.unique([f.split('_')[-2] for f in filenames])
print(f'Found subjects: {subjects}')

np.random.seed(42) # For reproducibility
permutation = np.random.permutation(len(subjects))
train_subjects = subjects[permutation[:-1]]
test_subjects = subjects[permutation[-1:]]

print(f'Training on subjects: {train_subjects}')
print(f'Testing on subjects: {test_subjects}')

window_size = 60
window_shift = 1
train_dataset = IMUDataset([f for f in filenames if any(s == f.split('_')[-2] for s in train_subjects)], 
                           filter=False, 
                           window_size=window_size, 
                           window_shift=window_shift)
test_dataset = IMUDataset([f for f in filenames if any(s == f.split('_')[-2] for s in test_subjects)], 
                          filter=False, 
                          window_size=window_size, 
                          window_shift=window_shift)

def collate_fn(batch):
    longest_sample = max(batch, key=lambda x: len(x['time']))
    max_len = len(longest_sample['time']) # max_len
    padded_batch = []

    for sample in batch:
        padding_len = max_len - len(sample['time'])
        padded_sample = {}
        padded_sample['mocap'] = torch.cat([sample['mocap'],
                                            sample['mocap'][-1:].repeat(padding_len,1,1)]) # (max_len, 8, 3)
        acc = torch.cat([sample['imu'][:,[0,1]],
                         sample['imu'][-1:,[0,1]].repeat(padding_len,1,1)]) # (max_len, 2, 3)
        gyro = torch.cat([sample['imu'][:,[2,3]],
                          torch.zeros_like(sample['imu'][-1:,[2,3]]).repeat(padding_len,1,1)]) # (max_len, 2, 3)
        padded_sample['imu'] = torch.cat([acc, gyro], dim=1) # (max_len, 4, 3)
        padded_batch.append(padded_sample)

    return {'time': longest_sample['time'],
            'mocap': torch.stack([sample['mocap'] for sample in padded_batch]),
            'imu': torch.stack([sample['imu'] for sample in padded_batch])}

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

def compute_mpjpe(predictions, targets):
    """
    predictions, targets: (B, T, J, 3)
    Returns: scalar MPJPE (in meters)
    """
    error = torch.norm(predictions - targets, dim=-1)  # (B, T, J)
    return error.mean().item()

def compute_mpve(predictions, targets):
    """
    predictions, targets: (B, T, J, 3)
    Returns: scalar MPVE (in m/s)
    """
    pred_velocity = predictions[:, 1:, :, :] - predictions[:, :-1, :, :]  # (B, T-1, J, 3)
    target_velocity = targets[:, 1:, :, :] - targets[:, :-1, :, :]        # (B, T-1, J, 3)

    velocity_error = torch.norm(pred_velocity - target_velocity, dim=-1)  # (B, T-1, J)
    return velocity_error.mean().item()

# 1) Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Instantiate model with the same hyperparameters you used at train time
### MODEL OPTION 1: Baseline LSTM -------------------------------

model = SimpleNN(input_size=4*3, hidden_size=8*3, num_layers=2) 

### -------------------------------------------------------------


### MODEL OPTION 2: ConvTransformer -----------------------------

# model = ConvTransformer(
#     input_dim=12,
#     transformer_dim=64,
#     window_size=window_size,
#     nhead=8,
#     dim_feedforward=256,
#     transformer_dropout=0.1,
#     transformer_activation="gelu",
#     num_encoder_layers=6,
#     encode_position=True
# )

###---------------------------------------------------------------


### MODEL OPTION 3: Bidirectional LSTM ---------------------------

# model = BiLSTM(input_size=12, hidden_size=12, num_layers=2)

###---------------------------------------------------------------
model.to(device)

# 3) Load pretrained weights
CHECKPOINT_PATH = "lstm_nopreprocess_withconstraints_batchsize32_winsize60_winshift15_new/checkpoint_epoch_24.pth"
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1} with loss {checkpoint['loss']:.4f}")

# 4) Evaluation loop
model.eval()
angle_weight = 0.1
bone_length_weight = 0.2

epoch_test_loss = 0
epoch_test_mse  = 0
epoch_test_angle = 0
epoch_test_bone  = 0
epoch_test_mpjpe = 0
epoch_test_mpve  = 0

with torch.no_grad():
    for i, batch in enumerate(test_dataloader, 1):
        # Move data to device
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Forward pass
        outputs = model(batch['imu'])  
        outputs_reshaped = outputs.view(*batch['mocap'].shape)  

        # Compute losses
        mse_loss   = F.mse_loss(outputs_reshaped, batch['mocap'])
        angle_loss = angle_constraint_loss(outputs_reshaped, alpha=angle_weight)
        bone_loss  = bone_length_consistency_loss(outputs_reshaped, alpha=bone_length_weight)
        total_loss = mse_loss + angle_loss + bone_loss

        # Accumulate
        epoch_test_loss  += total_loss.item()
        epoch_test_mse   += mse_loss.item()
        epoch_test_angle += angle_loss.item()
        epoch_test_bone  += bone_loss.item()
        epoch_test_mpjpe += compute_mpjpe(outputs_reshaped, batch['mocap'])
        epoch_test_mpve  += compute_mpve(outputs_reshaped, batch['mocap'])

        if i % 10 == 0:
            print(f'Test Step [{i}/{len(test_dataloader)}], '
                  f'Total Loss: {total_loss:.4f}, MSE: {mse_loss:.4f}, '
                  f'Angle: {angle_loss:.4f}, Bone: {bone_loss:.4f}')

# Compute averages
N = len(test_dataloader)
epoch_test_loss  /= N
epoch_test_mse   /= N
epoch_test_angle /= N
epoch_test_bone  /= N
epoch_test_mpjpe /= N
epoch_test_mpve  /= N

print(f'Final Evaluation → '
      f'Test Loss: {epoch_test_loss:.4f}, '
      f'MSE: {epoch_test_mse:.4f}, '
      f'MPJPE: {epoch_test_mpjpe:.4f}, '
      f'MPVE: {epoch_test_mpve:.4f}')

from utils import visualize_model_predictions

# Call the visualization function after training
visualize_model_predictions(model, test_dataloader)
