import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_skeleton_comparison(ground_truth, predictions, frame_idx=0, title='Skeleton Comparison'):
    """
    Visualize ground truth vs predicted skeletons
    
    Args:
        ground_truth: Tensor of shape (seq_len, 8, 3) or array-like
        predictions: Tensor of shape (seq_len, 8, 3) or array-like
        frame_idx: Index of the frame to visualize
        title: Plot title
    """
    # Convert to numpy if tensors
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Extract specific frame
    gt_frame = ground_truth[frame_idx]  # Shape: (8, 3)
    pred_frame = predictions[frame_idx]  # Shape: (8, 3)
    
    # Define the connections for visualization (joint indices)
    left_arm_connections = [(0, 1), (1, 2), (2, 3)]  # Shoulder -> Elbow -> Wrist -> Finger
    right_arm_connections = [(4, 5), (5, 6), (6, 7)]  # Shoulder -> Elbow -> Wrist -> Finger
    
    fig = plt.figure(figsize=(18, 6))
    
    # Setup 3 views: front (XZ), side (YZ), top (XY)
    views = [(0, 2, 'Front View (XZ)'), (1, 2, 'Side View (YZ)'), (0, 1, 'Top View (XY)')]
    
    for i, (dim1, dim2, view_title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_title(f"{view_title}")
        
        # Plot ground truth left arm
        for start_idx, end_idx in left_arm_connections:
            ax.plot([gt_frame[start_idx, dim1], gt_frame[end_idx, dim1]], 
                    [gt_frame[start_idx, dim2], gt_frame[end_idx, dim2]], 
                    'b-', linewidth=3, alpha=0.7)
        
        # Plot ground truth right arm
        for start_idx, end_idx in right_arm_connections:
            ax.plot([gt_frame[start_idx, dim1], gt_frame[end_idx, dim1]], 
                    [gt_frame[start_idx, dim2], gt_frame[end_idx, dim2]], 
                    'g-', linewidth=3, alpha=0.7)
        
        # Plot predicted left arm
        for start_idx, end_idx in left_arm_connections:
            ax.plot([pred_frame[start_idx, dim1], pred_frame[end_idx, dim1]], 
                    [pred_frame[start_idx, dim2], pred_frame[end_idx, dim2]], 
                    'b--', linewidth=2)
        
        # Plot predicted right arm
        for start_idx, end_idx in right_arm_connections:
            ax.plot([pred_frame[start_idx, dim1], pred_frame[end_idx, dim1]], 
                    [pred_frame[start_idx, dim2], pred_frame[end_idx, dim2]], 
                    'g--', linewidth=2)
        
        # Plot joints
        ax.scatter(gt_frame[:4, dim1], gt_frame[:4, dim2], c='blue', s=50, label='GT Left Arm')
        ax.scatter(gt_frame[4:, dim1], gt_frame[4:, dim2], c='green', s=50, label='GT Right Arm')
        ax.scatter(pred_frame[:4, dim1], pred_frame[:4, dim2], c='cyan', s=30, label='Pred Left Arm')
        ax.scatter(pred_frame[4:, dim1], pred_frame[4:, dim2], c='lime', s=30, label='Pred Right Arm')
        
        # Set the aspect ratio to be equal
        ax.set_aspect('equal')
        
        # Set labels
        ax.set_xlabel(f"Dimension {dim1}")
        ax.set_ylabel(f"Dimension {dim2}")
        
        # Add legend
        if i == 0:
            ax.legend(loc='upper left')  # or 'best' or 'lower left', etc.

        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    return fig

def create_skeleton_animation(ground_truth, predictions, save_path=None, fps=30, duration=None, frame_step=10):
    """
    Create animation comparing ground truth and predicted skeletons
    
    Args:
        ground_truth: Tensor of shape (seq_len, 8, 3) or array-like
        predictions: Tensor of shape (seq_len, 8, 3) or array-like
        save_path: Path to save the animation. If None, the animation is not saved
        fps: Frames per second
        duration: Duration of the animation in seconds. If None, all frames are used
    
    Returns:
        Animation object that can be displayed in a notebook
    """
    # Convert to numpy if tensors
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Define the connections for visualization
    left_arm_connections = [(0, 1), (1, 2), (2, 3)]  # Shoulder -> Elbow -> Wrist -> Finger
    right_arm_connections = [(4, 5), (5, 6), (6, 7)]  # Shoulder -> Elbow -> Wrist -> Finger
    
    # Calculate the total number of frames to display
    total_frames = len(ground_truth)
    if duration is not None:
        total_frames = min(total_frames, int(duration * fps))
    
    # Create figure and axes
    fig = plt.figure(figsize=(18, 6))
    axes = [fig.add_subplot(1, 3, i+1) for i in range(3)]
    
    # Set view titles
    view_titles = ['Front View (XZ)', 'Side View (YZ)', 'Top View (XY)']
    views = [(0, 2), (1, 2), (0, 1)]  # Dimensions to plot for each view
    
    for ax, title in zip(axes, view_titles):
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Set axis limits based on the data range with some padding
    pad = 0.1
    min_vals = np.min(np.concatenate([ground_truth, predictions]), axis=(0, 1))
    max_vals = np.max(np.concatenate([ground_truth, predictions]), axis=(0, 1))
    range_vals = max_vals - min_vals
    min_vals -= range_vals * pad
    max_vals += range_vals * pad
    
    for i, ax in enumerate(axes):
        dim1, dim2 = views[i]
        ax.set_xlim(min_vals[dim1], max_vals[dim1])
        ax.set_ylim(min_vals[dim2], max_vals[dim2])
    
    # Initialize lines and points for ground truth
    gt_left_lines = [[ax.plot([], [], 'b-', linewidth=3, alpha=0.7)[0] for _ in left_arm_connections] for ax in axes]
    gt_right_lines = [[ax.plot([], [], 'g-', linewidth=3, alpha=0.7)[0] for _ in right_arm_connections] for ax in axes]
    gt_left_points = [ax.scatter([], [], c='blue', s=50, label='GT Left Arm') for ax in axes]
    gt_right_points = [ax.scatter([], [], c='green', s=50, label='GT Right Arm') for ax in axes]
    
    # Initialize lines and points for predictions
    pred_left_lines = [[ax.plot([], [], 'b--', linewidth=2)[0] for _ in left_arm_connections] for ax in axes]
    pred_right_lines = [[ax.plot([], [], 'g--', linewidth=2)[0] for _ in right_arm_connections] for ax in axes]
    pred_left_points = [ax.scatter([], [], c='cyan', s=30, label='Pred Left Arm') for ax in axes]
    pred_right_points = [ax.scatter([], [], c='lime', s=30, label='Pred Right Arm') for ax in axes]
    
    # Add legend to the first axis
    axes[0].legend(loc='upper right')
    
    # Frame counter text
    frame_text = fig.text(0.5, 0.95, '', ha='center')
    
    def init():
        """Initialize the animation"""
        for ax_gt_left_lines in gt_left_lines:
            for line in ax_gt_left_lines:
                line.set_data([], [])
        
        for ax_gt_right_lines in gt_right_lines:
            for line in ax_gt_right_lines:
                line.set_data([], [])
        
        for ax_pred_left_lines in pred_left_lines:
            for line in ax_pred_left_lines:
                line.set_data([], [])
        
        for ax_pred_right_lines in pred_right_lines:
            for line in ax_pred_right_lines:
                line.set_data([], [])
        
        for point in gt_left_points + gt_right_points + pred_left_points + pred_right_points:
            point.set_offsets(np.empty((0, 2)))
        
        frame_text.set_text('')
        
        return (sum(gt_left_lines, []) + sum(gt_right_lines, []) + 
                sum(pred_left_lines, []) + sum(pred_right_lines, []) + 
                gt_left_points + gt_right_points + pred_left_points + pred_right_points + 
                [frame_text])
    
    def update(frame_idx):
        """Update the animation for a specific frame"""
        gt_frame = ground_truth[frame_idx]
        pred_frame = predictions[frame_idx]
        
        # Update ground truth lines and points
        for i, ax in enumerate(axes):
            dim1, dim2 = views[i]
            
            # Update ground truth left arm
            for j, (start_idx, end_idx) in enumerate(left_arm_connections):
                gt_left_lines[i][j].set_data(
                    [gt_frame[start_idx, dim1], gt_frame[end_idx, dim1]],
                    [gt_frame[start_idx, dim2], gt_frame[end_idx, dim2]]
                )
            
            # Update ground truth right arm
            for j, (start_idx, end_idx) in enumerate(right_arm_connections):
                gt_right_lines[i][j].set_data(
                    [gt_frame[start_idx, dim1], gt_frame[end_idx, dim1]],
                    [gt_frame[start_idx, dim2], gt_frame[end_idx, dim2]]
                )
            
            # Update predicted left arm
            for j, (start_idx, end_idx) in enumerate(left_arm_connections):
                pred_left_lines[i][j].set_data(
                    [pred_frame[start_idx, dim1], pred_frame[end_idx, dim1]],
                    [pred_frame[start_idx, dim2], pred_frame[end_idx, dim2]]
                )
            
            # Update predicted right arm
            for j, (start_idx, end_idx) in enumerate(right_arm_connections):
                pred_right_lines[i][j].set_data(
                    [pred_frame[start_idx, dim1], pred_frame[end_idx, dim1]],
                    [pred_frame[start_idx, dim2], pred_frame[end_idx, dim2]]
                )
            
            # Update points
            gt_left_points[i].set_offsets(gt_frame[:4, [dim1, dim2]])
            gt_right_points[i].set_offsets(gt_frame[4:, [dim1, dim2]])
            pred_left_points[i].set_offsets(pred_frame[:4, [dim1, dim2]])
            pred_right_points[i].set_offsets(pred_frame[4:, [dim1, dim2]])
        
        # Update frame counter
        frame_text.set_text(f'Frame: {frame_idx+1}/{total_frames}')
        
        return (sum(gt_left_lines, []) + sum(gt_right_lines, []) + 
                sum(pred_left_lines, []) + sum(pred_right_lines, []) + 
                gt_left_points + gt_right_points + pred_left_points + pred_right_points + 
                [frame_text])
    
    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init, 
        blit=True, interval=1000/fps, repeat=True
    )
    
    # Save animation if a path is provided
    if save_path is not None:
        # writer = animation.FFMpegWriter(fps=fps)
        # ani.save(save_path, writer=writer)
        # print(f"Animation saved to {save_path}")

        # Save .mp4
        mp4_writer = animation.FFMpegWriter(fps=fps)
        ani.save(save_path + '.mp4', writer=mp4_writer)
        
        # Save .gif
        gif_writer = animation.PillowWriter(fps=fps)
        ani.save(save_path + '.gif', writer=gif_writer)
    
    plt.close()
    return ani

def visualize_model_predictions(model, test_dataloader, num_samples=1, save_animations=False):
    """
    Visualize model predictions for a few test samples in a controlled manner.
    
    This updated version shows:
      - Only one sample per batch (as set by num_samples).
      - Two static frames: the first and the last frame.
      - One animation per sample showing the full sequence.
    
    For each sample, the function produces:
      1. Two static visualization figures with three subplots each (Front View, Side View, and Top View),
         comparing the ground truth and predicted skeletons.
      2. An animation that displays the entire motion sequence for that sample.
    """
    model.eval()
    
    with torch.no_grad():
        # Process only the first batch to reduce computation
        for i, batch in enumerate(test_dataloader):
            #Move batch data onto device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Run the model on the input IMU data and reshape prediction to match mocap data shape
            outputs = model(batch['imu'])
            outputs_reshaped = outputs.view(*batch['mocap'].shape)
            
            # For the static visualization and animation, process only the first 'num_samples'
            for j in range(min(num_samples, outputs_reshaped.size(0))):
                # Select two key frames: the first frame and the last frame
                frame_indices = [0, outputs_reshaped.size(1)-1]
                for frame_idx in frame_indices:
                    # Visualize static comparison for the selected frames
                    fig = visualize_skeleton_comparison(
                        batch['mocap'][j], outputs_reshaped[j], 
                        frame_idx=frame_idx, 
                        title=f'Sample {j+1}, Frame {frame_idx}'
                    )
                    # Save figure if needed, or display it
                    print("Saved figure for sample", j+1, "at frame", frame_idx)
                    plt.savefig(f'./result/skeleton_comparison_sample_{j+1}_frame_{frame_idx}.png')
                    plt.close(fig)
                
                # Create an animation for the full motion sequence of this sample.
                # The animation uses a specified frame rate (fps) and shows all frames.
                ani = create_skeleton_animation(
                    batch['mocap'][j], outputs_reshaped[j],
                    save_path=None,  # Set this to a file path if you wish to save the animation
                    fps=30
                )
                # If you are using a Jupyter Notebook, you can display the animation using:
                from IPython.display import HTML
                display(HTML(ani.to_jshtml()))
            
            # Process only the first batch to avoid visualizing too much data
            if i == 10:
                break

def visualize_each_file(model, dataset, device, fps=30, save_path = 'result'):
    model.eval()
    for fid, npz_path in enumerate(dataset.filenames):
        # 1) reconstruct only windows for file `fid`
        full_gt, full_pred = None, None
        with torch.no_grad():
            for win_fid, start in dataset._index:
                if win_fid != fid:
                    continue
                sample    = dataset._files[win_fid]
                imu_win   = sample['imu'][start : start+dataset.window_size][None]
                mocap_win = sample['mocap'][start : start+dataset.window_size]

                imu_t   = torch.from_numpy(imu_win).float().to(device)
                pred    = model(imu_t)[0].view(dataset.window_size, 8, 3).cpu().numpy()

                if full_gt is None:
                    full_gt   = mocap_win.copy()
                    full_pred = pred.copy()
                else:
                    full_gt   = np.concatenate([full_gt,   mocap_win[-1:]], axis=0)
                    full_pred = np.concatenate([full_pred, pred[-1:]],      axis=0)

        # 2) static snapshots
        for frame_idx in [0, full_gt.shape[0] - 1]:
            _ = visualize_skeleton_comparison(full_gt, full_pred,
                                              frame_idx=frame_idx,
                                              title=f"{Path(npz_path).stem} frame {frame_idx}")
            plt.show()

        # 3) animation + save
        stem    = Path(npz_path).stem
        base    = str(Path(save_path)/stem)
        ani = create_skeleton_animation(full_gt, full_pred,
                                        fps=fps, duration=None,
                                        frame_step=1, save_path=base)

