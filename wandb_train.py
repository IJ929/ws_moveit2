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
import logging

# W&B: Import the library
import wandb

# --- 1. Configuration ---
# Ensure the dataset file exists. If not, create a dummy one for demonstration.
DATASET_PATH = 'panda_ik_dataset.csv'
if not os.path.exists(DATASET_PATH):
    logging.warning(f"'{DATASET_PATH}' not found. Creating a dummy dataset for demonstration purposes.")
    num_dummy_samples = 1000
    dummy_data = {
        **{f'joint{i+1}': np.random.randn(num_dummy_samples) for i in range(7)},
        'pos_x': np.random.randn(num_dummy_samples), 'pos_y': np.random.randn(num_dummy_samples), 'pos_z': np.random.randn(num_dummy_samples),
        **{f'quat_{axis}': np.random.rand(num_dummy_samples) * 2 - 1 for axis in ['x', 'y', 'z', 'w']}
    }
    quat = np.array([dummy_data['quat_x'], dummy_data['quat_y'], dummy_data['quat_z'], dummy_data['quat_w']]).T
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    dummy_data['quat_x'], dummy_data['quat_y'], dummy_data['quat_z'], dummy_data['quat_w'] = quat.T
    pd.DataFrame(dummy_data).to_csv(DATASET_PATH, index=False)
    logging.info("Dummy dataset created.")

# Hyperparameters are now defined in the sweep config, but we keep defaults
INPUT_DIM = 7
NUM_EPOCHS = 50 # You can also sweep this if you like

# --- 2. Custom PyTorch Dataset ---
class IKDataset(Dataset):
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
    def __init__(self, input_dim, cond_dim, num_transforms, spline_bins):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
        transforms = []
        for i in range(num_transforms):
            coupling_layer = T.ConditionalPiecewiseRationalQuadraticCoupling(
                input_dim=input_dim, context_dim=cond_dim,
                hidden_dims=[256, 256], num_bins=spline_bins,
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
def train(model, dataloader, optimizer, scheduler, device, epochs, config):
    logging.info("Starting training run...")
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    accumulation_steps = config.get('accumulation_steps', 1) # Get from config

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad() # Move zero_grad() outside the loop
        
        for i, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                log_prob = model(x_batch, y_batch)
                loss = -log_prob.mean()
                loss = loss / accumulation_steps # Normalize loss

            scaler.scale(loss).backward()

            # Perform optimizer step only after accumulating gradients
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Reset gradients for the next accumulation cycle

            total_loss += loss.item() * accumulation_steps # Un-normalize for logging
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # W&B: Log metrics to W&B dashboard for every epoch
        wandb.log({"avg_loss": avg_loss, "epoch": epoch})
        
        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    logging.info("Training finished for this run.")


# W&B: Create a main function that the W&B agent can call
def run_sweep():
    # W&B: Initialize a new run
    with wandb.init() as run:
        # W&B: Get hyperparameters from the sweep config
        config = run.config
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        dataset = IKDataset(DATASET_PATH)
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, # W&B: Use config value
            shuffle=True,
            num_workers=os.cpu_count() // 2,
            pin_memory=True
        )
        
        model = ConditionalNormalizingFlow(
            input_dim=INPUT_DIM, 
            cond_dim=dataset.y_data.shape[1],
            num_transforms=config.num_transforms, # W&B: Use config value
            spline_bins=config.spline_bins        # W&B: Use config value
        ).to(device)
        
        # Compile the model if available (optional but recommended)
        try:
            model = torch.compile(model)
            logging.info("Model compiled successfully.")
        except Exception:
            logging.warning("torch.compile() is not available on this system. Running uncompiled.")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # W&B: Use config value
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=False)

        train(model, dataloader, optimizer, scheduler, device, epochs=NUM_EPOCHS, config=config)
        
        # W&B: Save the model file as an artifact to W&B
        model_path = "model.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)
        logging.info("Model saved and logged to W&B.")

# --- 5. Main Execution ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # W&B: Define the sweep configuration using a Python dictionary
    sweep_config = {
        'method': 'bayes',  # Use Bayesian Optimization
        'metric': {
            'name': 'avg_loss',
            'goal': 'minimize'   # We want to minimize the loss
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'batch_size': {
                'values': [256, 512, 1024]
            },
            'accumulation_steps': { 
                'values': [1, 2, 4] 
            },
            'num_transforms': {
                'distribution': 'int_uniform',
                'min': 4,
                'max': 12
            },
            'spline_bins': {
                'distribution': 'int_uniform',
                'min': 4,
                'max': 16
            }
        }
    }

    # W&B: Initialize the sweep
    # You can name your project whatever you like
    sweep_id = wandb.sweep(sweep_config, project="panda-ik-solver-sweep")

    # W&B: Start the sweep agent. It will run N times (e.g., 20)
    # The agent will call the 'run_sweep' function for each run.
    logging.info(f"Starting W&B sweep agent with ID: {sweep_id}")
    wandb.agent(sweep_id, function=run_sweep, count=20)
    
    logging.info("W&B sweep complete.")