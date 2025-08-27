# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pyro.distributions as dist
import pyro.distributions.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import logging
from pin_ik import RobotModel
from tqdm import tqdm

# --- Configuration ---
class Config:
    DATASET_PATH = 'data/panda_arm_training_data_pinocchio.csv'
    INPUT_DIM = 7
    NUM_EPOCHS = 13
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-4
    LEARNING_RATE_DECAY = 8e-1
    NUM_TRANSFORMS = 7
    HIDDEN_DIMS = [1024, 1024, 1024]
    SPLINE_BINS = 8
    MODEL_PATH = 'models/conditional_nf_model.pth'
    BASE_DIST_SCALE = 0.5

class NumpyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit_transform(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0) + 1e-6  # prevent division by zero
        return (data - self.mean_) / self.std_
    
    def transform(self, data):
        return (data - self.mean_) / self.std_
    
    def inverse_transform(self, data):
        return data * self.std_ + self.mean_

class IKDataset(Dataset):
    """Dataset for loading Panda IK data."""
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        joint_cols = [f'joint{i+1}' for i in range(Config.INPUT_DIM)]
        pose_cols = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
        
        self.x_scaler = NumpyStandardScaler()
        self.y_scaler = NumpyStandardScaler()
        
        x_data = self.x_scaler.fit_transform(df[joint_cols].values.astype(np.float32))
        y_data = self.y_scaler.fit_transform(df[pose_cols].values.astype(np.float32))
        
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class ConditionalNormalizingFlow(nn.Module):
    """Conditional Normalizing Flow model using Spline transforms and Permutations."""
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        self.register_buffer('base_dist_loc', torch.zeros(input_dim))
        self.register_buffer('base_dist_scale', torch.ones(input_dim) * Config.BASE_DIST_SCALE)
        
        self.transform_modules = nn.ModuleList([
            T.conditional_spline_autoregressive(
                input_dim=input_dim,
                context_dim=cond_dim,
                hidden_dims=Config.HIDDEN_DIMS,
                count_bins=Config.SPLINE_BINS,
                bound=input_dim // 2,
                order='quadratic'
            ) for _ in range(Config.NUM_TRANSFORMS)
        ])
        
        # Store permutation buffers
        for i in range(Config.NUM_TRANSFORMS - 1):
            self.register_buffer(f'perm_buffer_{i}', torch.randperm(input_dim, dtype=torch.long))

    def _build_flow(self):
        """Build the flow distribution on the correct device."""
        transforms = []
        for i in range(Config.NUM_TRANSFORMS):
            transforms.append(self.transform_modules[i])
            if i < Config.NUM_TRANSFORMS - 1:
                perm_tensor = getattr(self, f'perm_buffer_{i}')
                transforms.append(T.Permute(perm_tensor))
            
        base_dist = dist.Normal(self.base_dist_loc, self.base_dist_scale)
        return dist.ConditionalTransformedDistribution(base_dist, transforms)

    def forward(self, x, y):
        flow = self._build_flow()
        return flow.condition(y).log_prob(x)

    def sample(self, y, num_samples=1):
        with torch.no_grad():
            flow = self._build_flow()
            return flow.condition(y).sample(torch.Size([num_samples]))

def save_model_if_better(model, current_loss, best_loss, path):
    """Save model if current loss is better than best loss."""
    if current_loss < best_loss:
        torch.save(model.state_dict(), path)
        logging.info(f"Model saved to {path} with loss: {current_loss:.2e}")
        return current_loss
    return best_loss

def train(model, dataloader, optimizer, scheduler, device, robot):
    """Main training loop."""
    logging.info("Starting training...")
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_loss = float('inf')
    total_loss = 0
    n_steps = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        for x_batch, y_batch in pbar:
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
            n_steps += 1
            
            if n_steps > 50:
                avg_loss = total_loss / n_steps
                scheduler.step(avg_loss)
                best_loss = save_model_if_better(model, avg_loss, best_loss, Config.MODEL_PATH)
                err_norm = test_ik_solutions(model, robot, dataset, 100, 30)
                pbar.set_postfix({'Loss': f"{avg_loss:.2e}", 'Err': f"{err_norm:.2e}"})
                total_loss = n_steps = 0
                model.train()
    
    logging.info("Training finished.")

def generate_ik_solutions(model, pose, dataset, device, num_solutions=10):
    """Generate IK solutions for a given pose."""
    model.eval()
    pose_normalized = dataset.y_scaler.transform(pose.reshape(1, -1))
    pose_tensor = torch.from_numpy(pose_normalized.astype(np.float32)).to(device)
    joint_samples = model.sample(pose_tensor, num_samples=num_solutions)
    return dataset.x_scaler.inverse_transform(joint_samples.cpu().numpy())

def test_ik_solutions(model, robot, dataset, num_samples=10, num_solutions=10):
    """Test IK solutions generated by the model."""
    model.eval()
    random_configs = robot.sample_random_configurations(num_samples)
    errors = []
    
    for config in random_configs:
        target_pose_se3 = robot.forward_kinematics(config)
        target_pose = robot.convert_se3_to_pose(target_pose_se3)
        ik_solutions = generate_ik_solutions(model, target_pose, dataset, device, num_solutions)
        
        for sol in ik_solutions:
            current_pose = robot.forward_kinematics(sol)
            err = robot.compute_se3_error(current_pose, target_pose_se3)
            errors.append(np.linalg.norm(err))
    
    avg_error = np.mean(errors)
    logging.info(f"Average IK error: {avg_error:.2e} m")
    return avg_error

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset = IKDataset(Config.DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = ConditionalNormalizingFlow(Config.INPUT_DIM, dataset.y_data.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.LEARNING_RATE_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    robot = RobotModel('panda', joints_to_lock_names=['panda_finger_joint1', 'panda_finger_joint2'])

    # Initial test
    test_ik_solutions(model, robot, dataset, 10, 10)

    # Training
    train(model, dataloader, optimizer, scheduler, device, robot)

    # Demonstration
    logging.info("--- IK Solver Demonstration ---")
    original_df = pd.read_csv(Config.DATASET_PATH)
    pose_cols = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    test_pose = original_df[pose_cols].iloc[42].values
    
    logging.info(f"Target pose: {test_pose}")
    
    start_time = time.time()
    ik_solutions = generate_ik_solutions(model, test_pose, dataset, device, 5)
    end_time = time.time()
    
    logging.info(f"Generated 5 solutions in {end_time - start_time:.6f} seconds:")
    for i, sol in enumerate(ik_solutions):
        logging.info(f"Solution {i+1}: {np.round(sol, 4)}")

    logging.info("Complete.")
# %%