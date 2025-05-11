import torch
import torch.nn.functional as F

EPS = 1e-6 # For Numerical Stability

### ⭐ Bio-mechanical onstraints

def angle_constraint_loss(predictions, alpha=0.1, min_angle=30, max_angle=180):
    """
    Joint angle constraint loss.
    - Computes actual angles using arccos and penalizes any values that fall outside the natural range (min_angle, max_angle).
    - Angles are converted to degrees for comparison.
    """
    # Extract left/right arm joints from predictions with shape (batch_size, seq_len, 8, 3)
    # Joint order is assumed as:
    # [left_shoulder, left_elbow, left_wrist, left_finger, right_shoulder, right_elbow, right_wrist, right_finger]
    left_shoulder = predictions[:, :, 0, :]
    left_elbow = predictions[:, :, 1, :]
    left_wrist = predictions[:, :, 2, :]
    
    right_shoulder = predictions[:, :, 4, :]
    right_elbow = predictions[:, :, 5, :]
    right_wrist = predictions[:, :, 6, :]
    
    # Compute vectors between joints
    left_upper = left_elbow - left_shoulder
    left_fore = left_wrist - left_elbow
    right_upper = right_elbow - right_shoulder
    right_fore = right_wrist - right_elbow
    
    # Normalize the vectors
    left_upper_norm = eps_norm(left_upper)
    left_fore_norm = eps_norm(left_fore)
    right_upper_norm = eps_norm(right_upper)
    right_fore_norm = eps_norm(right_fore)
    
    # Compute cosine similarity and clamp for numerical stability
    left_cosine = torch.clamp(torch.sum(left_upper_norm * left_fore_norm, dim=-1), -1.0, 1.0)
    right_cosine = torch.clamp(torch.sum(right_upper_norm * right_fore_norm, dim=-1), -1.0, 1.0)
    
    # Convert angles from radians to degrees
    left_angles = torch.acos(left_cosine) * 180.0 / torch.pi
    right_angles = torch.acos(right_cosine) * 180.0 / torch.pi
    
    # Compute penalty for values outside the allowable range
    left_penalty = torch.clamp(left_angles - max_angle, min=0) + torch.clamp(min_angle - left_angles, min=0)
    right_penalty = torch.clamp(right_angles - max_angle, min=0) + torch.clamp(min_angle - right_angles, min=0)
    
    # Average penalty across batch and sequence
    total_penalty = torch.mean(left_penalty) + torch.mean(right_penalty)
    
    return alpha * total_penalty

def eps_norm(v):
    return v / (v.norm(p=2,dim=-1,keepdim=True) + EPS)

def bone_length_consistency_loss(predictions, alpha=0.1):
    """
    Calculate loss for bone length consistency
    Ensures bone lengths remain constant over time
    """
    # Extract the joint positions
    left_shoulders = predictions[:, :, 0, :]
    left_elbows = predictions[:, :, 1, :]
    left_wrists = predictions[:, :, 2, :]
    left_fingers = predictions[:, :, 3, :]
    
    right_shoulders = predictions[:, :, 4, :]
    right_elbows = predictions[:, :, 5, :]
    right_wrists = predictions[:, :, 6, :]
    right_fingers = predictions[:, :, 7, :]
    
    # Calculate bone vectors
    left_upper_arm = left_elbows - left_shoulders
    left_forearm = left_wrists - left_elbows
    left_hand = left_fingers - left_wrists
    
    right_upper_arm = right_elbows - right_shoulders
    right_forearm = right_wrists - right_elbows  
    right_hand = right_fingers - right_wrists
    
    # Calculate bone lengths at each time step
    left_upper_arm_lengths = torch.sqrt(torch.sum(left_upper_arm**2, dim=-1) + EPS)
    left_forearm_lengths = torch.sqrt(torch.sum(left_forearm**2, dim=-1) + EPS)
    left_hand_lengths = torch.sqrt(torch.sum(left_hand**2, dim=-1) + EPS)
    
    right_upper_arm_lengths = torch.sqrt(torch.sum(right_upper_arm**2, dim=-1) + EPS)
    right_forearm_lengths = torch.sqrt(torch.sum(right_forearm**2, dim=-1) + EPS)
    right_hand_lengths = torch.sqrt(torch.sum(right_hand**2, dim=-1) + EPS)
    
    # Calculate variance of bone lengths over time (for each sequence in batch)
    left_upper_arm_var = torch.var(left_upper_arm_lengths, dim=1)
    left_forearm_var = torch.var(left_forearm_lengths, dim=1)
    left_hand_var = torch.var(left_hand_lengths, dim=1)
    
    right_upper_arm_var = torch.var(right_upper_arm_lengths, dim=1)
    right_forearm_var = torch.var(right_forearm_lengths, dim=1)
    right_hand_var = torch.var(right_hand_lengths, dim=1)
    
    # Calculate total variance (average across batch)
    total_var = (torch.mean(left_upper_arm_var) + 
                torch.mean(left_forearm_var) + 
                torch.mean(left_hand_var) +
                torch.mean(right_upper_arm_var) + 
                torch.mean(right_forearm_var) + 
                torch.mean(right_hand_var))
    
    return alpha * total_var