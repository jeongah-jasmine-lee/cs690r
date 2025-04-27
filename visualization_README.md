# Skeleton Prediction Visualization

This module visualizes and compares **ground truth** and **predicted skeletons** for 3D motion data.  
It provides **static plots** and **animated sequences** to evaluate model performance in human motion prediction tasks.

## Functions Overview

- `visualize_skeleton_comparison`:  
  Generates static skeleton plots at a specific frame (e.g., start and end frame).

- `create_skeleton_animation`:  
  Animates the full motion sequence comparing ground truth and prediction over time.

- `visualize_model_predictions`:  
  Runs the model on test data and creates static plots and animations for selected samples.

## How It Works

1. **Input:**  
   - IMU data (accelerometer + gyroscope signals) from the test set.
   - Trained model predicting 3D joint positions (8 joints × 3D).

2. **Processing:**  
   - The model predicts the skeleton sequence.
   - Static skeletons are visualized at two frames (first and last frame).
   - An animation is created showing motion over time.

3. **Output:**  
   - Static plots saved as `.png` files under the `./result/` directory.
   - Animated skeleton sequences displayed (and optionally saved).

## Current Behavior

- Only the **first batch** from the test dataset is visualized to save time during initial experiments.
- Static comparison figures are saved automatically.
- Animations are displayed inline when using Jupyter Notebook.


## Final Evaluation Instructions

Once you have selected the **best model**, you should visualize the entire test dataset to report reliable results.

- **How to update:**  
  Remove or comment out the `break` statement inside `visualize_model_predictions` function.
