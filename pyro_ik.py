import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import logging # Import the logging module

# --- 1. Configuration ---
# Ensure the dataset file exists. If not, create a dummy one for demonstration.
DATASET_PATH = 'panda_ik_dataset.csv'
if not os.path.exists(DATASET_PATH):
    # Use logging.warning for non-critical alerts
    logging.warning(f"'{DATASET_PATH}' not found. Creating a dummy dataset for demonstration purposes.")
    num_dummy_samples = 1000
    dummy_data = {
        **{f'joint{i+1}': np.random.randn(num_dummy_samples) for i in range(7)},
        'pos_x': np.random.randn(num_dummy_samples),
        'pos_y': np.random.randn(num_dummy_samples),
        'pos_z': np.random.randn(num_dummy_samples),
        **{f'quat_{axis}': np.random.rand(num_dummy_samples) * 2 - 1 for axis in ['x', 'y', 'z', 'w']}
    }
    # Normalize quaternions
    quat = np.array([dummy_data['quat_x'], dummy_data['quat_y'], dummy_data['quat_z'], dummy_data['quat_w']]).T
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    dummy_data['quat_x'], dummy_data['quat_y'], dummy_data['quat_z'], dummy_data['quat_w'] = quat.T
    
    pd.DataFrame(dummy_data).to_csv(DATASET_PATH, index=False)
    logging.info("Dummy dataset created.")


# Hyperparameters
INPUT_DIM = 7   # 7 joints for Panda arm
NUM_EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
NUM_TRANSFORMS = 8
SPLINE_BINS = 8

# --- 2. Custom PyTorch Dataset ---
class IKDataset(Dataset):
    """Dataset for loading Panda IK data."""
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        joint_cols = [f'joint{i+1}' for i in range(INPUT_DIM)]
        pose_cols = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
        self.x_data = df[joint_cols].values.astype(np.float32)
        self.y_data = df[pose_cols].values.astype(np.float32)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
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
        transforms = []
        for i in range(num_transforms):
            coupling_layer = T.ConditionalPiecewiseRationalQuadraticCoupling(
                input_dim=input_dim,
                context_dim=cond_dim,
                hidden_dims=[256, 256],
                num_bins=spline_bins,
                split_dim=input_dim // 2
            )
            transforms.append(coupling_layer)
            if i < num_transforms - 1:
                transforms.append(T.Permute(torch.randperm(input_dim, dtype=torch.long)))
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
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
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

    dataset = IKDataset(DATASET_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )
    
    model = ConditionalNormalizingFlow(
        input_dim=INPUT_DIM, 
        cond_dim=dataset.y_data.shape[1],
        num_transforms=NUM_TRANSFORMS,
        spline_bins=SPLINE_BINS
    ).to(device)
    
    # Compile the model for a significant speedup on compatible hardware
    logging.info("Compiling the model with torch.compile()... (This may take a moment)")
    model = torch.compile(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    train(model, dataloader, optimizer, scheduler, device, epochs=NUM_EPOCHS)

    # --- DEMONSTRATION OF INFERENCE ---
    logging.info("--- IK Solver Demonstration ---")
    
    original_df = pd.read_csv(DATASET_PATH)
    pose_cols = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    test_pose = original_df[pose_cols].iloc[42].values
    
    logging.info(f"Target End-Effector Pose:\n{test_pose}")
    
    start_time = time.time()
    ik_solutions = generate_ik_solutions(model, test_pose, dataset, device, num_solutions=5)
    end_time = time.time()
    
    logging.info(f"Generated 5 diverse joint configurations in {end_time - start_time:.6f} seconds:")
    for i, sol in enumerate(ik_solutions):
        logging.info(f"Solution {i+1}: {np.round(sol, 4)}")

    logging.info("Model training and demonstration complete.")