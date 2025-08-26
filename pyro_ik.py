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
import logging # Import the logging module

# --- 1. Configuration ---
# Ensure the dataset file exists. If not, create a dummy one for demonstration.
DATASET_PATH = 'data/panda_arm_training_data.csv'
# Hyperparameters
INPUT_DIM = 7   # 7 joints for Panda arm
NUM_EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
NUM_TRANSFORMS = 8
SPLINE_BINS = 8

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

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x_data[idx]), torch.from_numpy(self.y_data[idx])
    
# --- 3. Advanced Conditional Normalizing Flow Model ---
class ConditionalNormalizingFlow(nn.Module):
    """An advanced Conditional Normalizing Flow model using Spline transforms and Permutations."""
    def __init__(self, input_dim, cond_dim, num_transforms, spline_bins):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
        
        # 1. Build the full sequence of transforms in a standard Python list.
        transforms = []
        for i in range(num_transforms):
            # The spline autoregressive transform IS an nn.Module
            transforms.append(T.conditional_spline_autoregressive(
                input_dim=input_dim,
                context_dim=cond_dim,
                hidden_dims=[256, 256],
                count_bins=spline_bins,
                bound=input_dim // 2
            ))
            # The Permute transform is NOT an nn.Module, so we handle its tensor manually.
            if i < num_transforms - 1:
                # 1. Create the permutation tensor.
                perm_tensor = torch.randperm(input_dim, dtype=torch.long)
                
                # 2. Register it as a buffer. This is the key change.
                #    The buffer will be automatically moved to the correct device
                #    when you call .to(device) on the model.
                buffer_name = f'perm_buffer_{i}'
                self.register_buffer(buffer_name, perm_tensor)
                
                # 3. Create the Permute transform using the registered buffer.
                #    We retrieve it using getattr().
                transforms.append(T.Permute(getattr(self, buffer_name)))

        # 2. Create the nn.ModuleList for PyTorch by filtering the list from step 1.
        # This uses a list comprehension to find all objects that are instances of nn.Module.
        self.transform_modules = nn.ModuleList([t for t in transforms if isinstance(t, nn.Module)])

        # Now, create the distribution using the registered ModuleList
        self.flow = dist.ConditionalTransformedDistribution(self.base_dist, transforms)

    def forward(self, x, y):
        conditioned_flow = self.flow.condition(y)
        return conditioned_flow.log_prob(x)

    def sample(self, y, num_samples=1):
        with torch.no_grad():
            conditioned_flow = self.flow.condition(y)
            return conditioned_flow.sample(torch.Size([num_samples]))

# --- 4. Advanced Training and Inference ---
def train(model, dataloader, optimizer, scheduler, device, epochs):
    """Main training loop with advanced features."""
    logging.info("Starting training with advanced methods...")
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
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
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
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
        num_workers=os.cpu_count() // 2,
        pin_memory=True
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

    # Compile the model for a significant speedup on compatible hardware
    logging.info("Compiling the model with torch.compile()... (This may take a moment)")
    model = torch.compile(model)

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