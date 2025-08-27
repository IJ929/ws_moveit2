# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import logging

from tqdm import tqdm, trange # Import the logging module

# --- 1. Configuration ---
# Ensure the dataset file exists. If not, create a dummy one for demonstration.
DATASET_PATH = 'data/panda_arm_training_data.csv'
# Hyperparameters
INPUT_DIM = 7   # 7 joints for Panda arm
NUM_EPOCHS = 13
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_TRANSFORMS = 7
SPLINE_BINS = 8
MODEL_PATH = 'models/conditional_nf_model.pth'

class NumpyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit_transform(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        return (data - self.mean_) / self.std_
    
    def inverse_transform(self, data):
        return data * self.std_ + self.mean_

# --- 2. Custom PyTorch Dataset ---
class IKDataset(Dataset):
    """Dataset for loading Panda IK data."""
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        joint_cols = [f'joint{i+1}' for i in range(INPUT_DIM)]
        pose_cols = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
        self.x_data = df[joint_cols].values.astype(np.float32)
        self.y_data = df[pose_cols].values.astype(np.float32)
        self.x_scaler = NumpyStandardScaler()
        self.y_scaler = NumpyStandardScaler()
        self.x_data = self.x_scaler.fit_transform(self.x_data)
        self.y_data = self.y_scaler.fit_transform(self.y_data)
        
        # move data to GPU if available
        if torch.cuda.is_available():
            self.x_data = torch.from_numpy(self.x_data).to('cuda')
            self.y_data = torch.from_numpy(self.y_data).to('cuda')

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# --- 3. Advanced Conditional Normalizing Flow Model ---
class ConditionalNormalizingFlow(nn.Module):
    """An advanced Conditional Normalizing Flow model using Spline transforms and Permutations."""
    def __init__(self, input_dim, cond_dim, num_transforms, spline_bins):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.num_transforms = num_transforms

        # Register base distribution parameters as buffers to ensure they are moved to the correct device
        self.register_buffer('base_dist_loc', torch.zeros(input_dim))
        self.register_buffer('base_dist_scale', torch.ones(input_dim))
        
        # Store the learnable (nn.Module) transforms and the permutation tensors (as buffers)
        self.transform_modules = nn.ModuleList()
        for i in range(num_transforms):
            # The spline autoregressive transform is an nn.Module
            self.transform_modules.append(T.conditional_spline_autoregressive(
                input_dim=input_dim,
                context_dim=cond_dim,
                hidden_dims=[256, 256],
                count_bins=spline_bins,
                bound=input_dim // 2
            ))
            # For the Permute transform, we only store its permutation tensor as a buffer
            if i < num_transforms - 1:
                buffer_name = f'perm_buffer_{i}'
                self.register_buffer(buffer_name, torch.randperm(input_dim, dtype=torch.long))

    def _build_flow(self):
        """
        Dynamically builds the list of transforms and the distribution on the correct device.
        This is called during the forward pass to ensure device consistency.
        """
        transforms = []
        for i in range(self.num_transforms):
            # Append the spline module from our stored list
            transforms.append(self.transform_modules[i])
            
            # Create and append a *new* Permute transform using the buffer, which is now on the correct device
            if i < self.num_transforms - 1:
                buffer_name = f'perm_buffer_{i}'
                perm_tensor = getattr(self, buffer_name)
                transforms.append(T.Permute(perm_tensor))
            
        # Create the base distribution using the buffers, which are now on the correct device
        base_dist = dist.Normal(self.base_dist_loc, self.base_dist_scale)
        # Return the complete, device-aware distribution
        return dist.ConditionalTransformedDistribution(base_dist, transforms)

    def forward(self, x, y):
        # Lazily build the flow distribution on the correct device
        flow = self._build_flow()
        conditioned_flow = flow.condition(y)
        return conditioned_flow.log_prob(x)

    def sample(self, y, num_samples=1):
        with torch.no_grad():
            # Lazily build the flow distribution on the correct device
            flow = self._build_flow()
            conditioned_flow = flow.condition(y)
            return conditioned_flow.sample(torch.Size([num_samples]))
        
def save_model(model, path):
    """Saves the model state dictionary to the specified path."""
    model.eval()
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")
    
def save_if_better(model, current_loss, best_loss, path):
    """Saves the model if the current loss is better than the best loss."""
    if current_loss < best_loss:
        save_model(model, path)
        return current_loss
    return best_loss
    
def load_model(model, path, device):
    """Loads the model state dictionary from the specified path."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    logging.info(f"Model loaded from {path}")

# --- 4. Advanced Training and Inference ---
def train(model, dataloader, optimizer, scheduler, device, epochs):
    """Main training loop with advanced features."""
    logging.info("Starting training with advanced methods...")
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        n_steps = 0
        process_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for x_batch, y_batch in process_bar:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                log_prob = model(x_batch, y_batch)
                loss = -log_prob.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            n_steps += 1
            if n_steps > 2_000:
                avg_loss = total_loss / n_steps
                scheduler.step(avg_loss)
                process_bar.set_postfix({'Avg Loss': f"{avg_loss:.2e}"})
                total_loss = 0
                n_steps = 0
                best_loss = save_if_better(model, avg_loss, best_loss, MODEL_PATH)
                model.train()
    logging.info("Training finished.")

def generate_ik_solutions(model, pose, dataset, device, num_solutions=10):
    """Generates IK solutions for a given pose using the trained model."""
    model.eval()
    pose_normalized = dataset.y_scaler.transform(pose.reshape(1, -1))
    pose_tensor = torch.from_numpy(pose_normalized.astype(np.float32)).to(device)
    joint_samples_normalized = model.sample(pose_tensor, num_samples=num_solutions)
    joint_samples_unscaled = dataset.x_scaler.inverse_transform(joint_samples_normalized.cpu().numpy())
    return joint_samples_unscaled

# %% 
# --- 5. Main Execution ---
if __name__ == '__main__':
    # --- Configure Logging ---
    # This sets up logging to output to the console with a timestamp and log level.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()] 
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # %%
    dataset = IKDataset(DATASET_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
    )
    
    # %%
    model = ConditionalNormalizingFlow(
        input_dim=INPUT_DIM, 
        cond_dim=dataset.y_data.shape[1],
        num_transforms=NUM_TRANSFORMS,
        spline_bins=SPLINE_BINS
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    # %%
    train(model, dataloader, optimizer, scheduler, device, epochs=NUM_EPOCHS)

    # %%
    # --- DEMONSTRATION OF INFERENCE ---
    logging.info("--- IK Solver Demonstration ---")
    original_df = pd.read_csv(DATASET_PATH)
    pose_cols = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    test_pose = original_df[pose_cols].iloc[42].values
    
    logging.info(f"Target End-Effector Pose:\n{test_pose}")

    # %%
    start_time = time.time()
    ik_solutions = generate_ik_solutions(model, test_pose, dataset, device, num_solutions=5)
    end_time = time.time()
    
    logging.info(f"Generated 5 diverse joint configurations in {end_time - start_time:.6f} seconds:")
    for i, sol in enumerate(ik_solutions):
        logging.info(f"Solution {i+1}: {np.round(sol, 4)}")

    logging.info("Model training and demonstration complete.")
# %%