import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import transformers
import shutil
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim, nn
from sklearn.metrics import mean_squared_error
from transformers import get_cosine_schedule_with_warmup
from datetime import datetime
from models.utilities import *
from models.evaluation_FE import *



# Add model and module directories to system path
sys.path.append('/Users/josephpassant/DissertationRedo/Informer2020/models')
from jpformer import JPFormer  # Import GPFormer model




PROJECT_ROOT = "/Users/josephpassant/DissertationRedo"

# Load config
def load_config(config_path="/Users/josephpassant/DissertationRedo/Informer2020/models/base_training_config_FE.json"):
    # print(f"Attempting to load config from: {config_path}")  # Debugging print statement
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as file:
        return json.load(file)

class ConfigObject:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


config_dict = load_config()
config = ConfigObject(config_dict)


# Setup device
if torch.cuda.is_available() and config.use_gpu:
    device = torch.device(f"cuda:{config.gpu}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon (M1/M2)
    print("Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Model setup

model = JPFormer(
    enc_in=config.enc_in,
    dec_in=config.dec_in,
    c_out=config.c_out,
    seq_len=config.seq_len,
    label_len=config.label_len,
    out_len=config.pred_len,
    d_model=config.d_model,
    n_heads=config.n_heads,
    e_layers=config.e_layers,
    d_layers=config.d_layers,
    d_ff=config.d_ff,
    factor=config.factor,
    dropout=config.dropout,
    embed=config.embed,
    activation=config.activation,
    output_attention=config.output_attention,
    mix=config.mix,
    device=device
).float().to(device)

# Debugging: Check model name
model_name = model.__class__.__name__
print(f"Initialized model: {model_name}")
model_name = model.__class__.__name__


TRAIN_ENCODER_DIR = os.path.join(PROJECT_ROOT, "/Users/josephpassant/DissertationRedo/data/FeatureEnhancedProcessedData/Training/ReplaceTraining/EncoderSlices")
TRAIN_DECODER_DIR = os.path.join(PROJECT_ROOT, "/Users/josephpassant/DissertationRedo/data/FeatureEnhancedProcessedData/Training/ReplaceTraining/DecoderSlices")
TRAIN_TARGET_DIR = os.path.join(PROJECT_ROOT, "/Users/josephpassant/DissertationRedo/data/FeatureEnhancedProcessedData/Training/ReplaceTraining/TargetSlices")

VAL_ENCODER_DIR = os.path.join(PROJECT_ROOT, "/Users/josephpassant/DissertationRedo/data/FeatureEnhancedProcessedData/Validation/ReplaceValidation/EncoderSlices")
VAL_DECODER_DIR = os.path.join(PROJECT_ROOT, "/Users/josephpassant/DissertationRedo/data/FeatureEnhancedProcessedData/Validation/ReplaceValidation/DecoderSlices")
VAL_TARGET_DIR = os.path.join(PROJECT_ROOT, "/Users/josephpassant/DissertationRedo/data/FeatureEnhancedProcessedData/Validation/ReplaceValidation/TargetSlices")

# Define test dataset and DataLoader
TEST_ENCODER_DIR = os.path.join(PROJECT_ROOT, "/Users/josephpassant/DissertationRedo/data/FeatureEnhancedProcessedData/Testing/ReplaceTest/EncoderSlices")
TEST_DECODER_DIR = os.path.join(PROJECT_ROOT, "/Users/josephpassant/DissertationRedo/data/FeatureEnhancedProcessedData/Testing/ReplaceTest/DecoderSlices")
TEST_TARGET_DIR = os.path.join(PROJECT_ROOT, "/Users/josephpassant/DissertationRedo/data/FeatureEnhancedProcessedData/Testing/ReplaceTest/TargetSlices")

# Training Data
train_dataset = BloodGlucoseDataset(TRAIN_ENCODER_DIR, TRAIN_DECODER_DIR, TRAIN_TARGET_DIR)
val_dataset = BloodGlucoseDataset(VAL_ENCODER_DIR, VAL_DECODER_DIR, VAL_TARGET_DIR)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=config.num_workers)
train_iter = ForeverDataIterator(train_loader)

# Test Data
test_dataset = BloodGlucoseDataset(TEST_ENCODER_DIR, TEST_DECODER_DIR, TEST_TARGET_DIR)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=config.num_workers)

# Wrap training DataLoader for continuous iteration
train_iter = ForeverDataIterator(train_loader)

# Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
# optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.wd)
total_steps = len(train_loader) * config.train_epochs
# use basic cosine annealing scheduler
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

#lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)


def create_dir(directory):
    """
    Safely creates a directory if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print(f"Directory {directory} already exists.")



def validate(val_loader, model):
    """
    Runs validation and computes RMSE & MAE at each time step.
    """
    model.eval()
    total_rmse, total_mae = 0, 0
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for batch_x, batch_dec, batch_y in val_loader:
            batch_x, batch_dec, batch_y = batch_x.to(device), batch_dec.to(device), batch_y.to(device)

            # Get predictions from the model

            outputs = model(batch_x, batch_dec)

            # Ensure target shape matches predictions
            batch_y = batch_y.unsqueeze(-1) if batch_y.ndim == 2 else batch_y  

            # Compute per-batch RMSE & MAE
            batch_rmse = rmse(outputs, batch_y)
            batch_mae = mae(outputs, batch_y)

            total_rmse += batch_rmse
            total_mae += batch_mae

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    # Convert lists to numpy arrays
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute RMSE & MAE per time step
    rmse_per_timestep = np.sqrt(np.mean((all_outputs - all_targets) ** 2, axis=0))
    mae_per_timestep = np.mean(np.abs(all_outputs - all_targets), axis=0)

    # Compute overall RMSE & MAE
    avg_rmse = total_rmse / max(len(val_loader), 1)
    avg_mae = total_mae / max(len(val_loader), 1)

    print(f"Validation RMSE: {avg_rmse:.6f}")
    print(f"Validation MAE: {avg_mae:.6f}")

    print("\nPer-Time Step RMSE & MAE:")
    for i, (rmse_t, mae_t) in enumerate(zip(rmse_per_timestep.flatten(), mae_per_timestep.flatten())):
        print(f"Time Step {i+1}: RMSE={rmse_t:.6f}, MAE={mae_t:.6f}")

    return avg_rmse, avg_mae, rmse_per_timestep, mae_per_timestep



def train(train_iter, model, optimizer, lr_scheduler, config):
    """
    Trains the model using batches from the training set.
    Implements early stopping based on validation loss.
    """
    best_loss = float("inf")
    best_model_state_dict = None
    validation_loss_table = []

    for epoch in range(config.train_epochs):
        print(f"Epoch {epoch+1}/{config.train_epochs}")

        epoch_start_time = time.time()  

        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":6.6f")  # Higher precision
        progress = ProgressMeter(len(train_loader), [batch_time, losses], prefix=f"Epoch: [{epoch+1}]")

        model.train()
        epoch_loss = 0
        end = time.time()

        for batch_idx in range(config.iters_per_epoch):
            batch_x, batch_dec, batch_y = next(train_iter)
            batch_x, batch_dec, batch_y = batch_x.to(device), batch_dec.to(device), batch_y.to(device)

            optimizer.zero_grad()
            
            # print(f"batch_x shape: {batch_x.shape}")
            # print(f"batch_dec shape: {batch_dec.shape}")
            outputs = model(batch_x, batch_dec)

            # Ensure batch_y has the same shape as outputs
            batch_y = batch_y.unsqueeze(-1) if batch_y.ndim == 2 else batch_y  
            loss = rmse(outputs, batch_y)

            # Backward pass & optimization step
            loss.backward()
            optimizer.step()

            # Update batch tracking
            losses.update(loss.item(), batch_x.size(0))
            epoch_loss += loss.item()
            batch_time.update(time.time() - end)
            end = time.time()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                progress.display(batch_idx)

        lr_scheduler.step()
        # Ensure the last update to progress meter is displayed
        progress.display(len(train_loader) - 1)

        avg_epoch_loss = epoch_loss / config.iters_per_epoch
        val_rmse, val_mae, rmse_per_timestep, mae_per_timestep = validate(val_loader, model)

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1} | Epoch Train Loss: {avg_epoch_loss:.6f} | Validation RMSE: {val_rmse:.6f} | Validation MAE: {val_mae:.6f}")

        validation_loss_table.append({
            "Epoch": epoch+1,
            "ValRMSE": val_rmse,
            "ValMAE": val_mae,
            "Time": epoch_time
        })

        if val_rmse < best_loss:
            best_loss = val_rmse
            best_model_state_dict = model.state_dict() 
            print(f"New best model found at epoch {epoch+1} with RMSE: {val_rmse:.6f}")

        # if (epoch+1) % 5 == 0:
        #     print (f"\n-----------------Testing at epoch {epoch+1}-----------------")
        #     tester = ModelTester(model, test_loader, device)
        #     test_rmse, test_mae, _, _ = tester.test()
        #     print(f"Test RMSE: {test_rmse:.6f}")
        #     print(f"Test MAE: {test_mae:.6f}")
        #     print("---------------------------------------------------------\n")

    if best_model_state_dict:

        save_dir = os.path.join(PROJECT_ROOT, "SavedModels", "best_validation_model")
        create_dir(save_dir)

        best_model_path = os.path.join(save_dir, f"{model_type}_VAL_RMSE_{best_loss:.4f}.pth")
        torch.save(best_model_state_dict, best_model_path)
        print(f"Best model saved at {best_model_path} with RMSE: {best_loss:.6f}")
    else:
        best_model_path = None

    df_val_loss = pd.DataFrame(validation_loss_table)

    return best_model_path, df_val_loss



if __name__ == "__main__":

    model_type = "jpformer_FE"

    print("-------------Training-------------")
    best_model_path, validation_metrics_df = train(train_iter, model, optimizer, lr_scheduler, config)

    print("\n-------------Testing-------------")
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path} for final testing.")

    tester = ModelTester(model, test_loader, device)

    test_rmse, test_mae, stats_tables, detailed_df = tester.test()

    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_description = f"{model_type}_{test_rmse:.4f}_MAE_{test_mae:.4f}"
    save_dir = os.path.join(PROJECT_ROOT, "SavedModels", "test_results", f"{model_name}","base_initial_eval", f"{model_description}_{timestamp}")
    create_dir(save_dir)

    if not os.path.exists(save_dir):
        print(f"ERROR: Directory {save_dir} was not created!")
        exit(1)

    config_path = "/Users/josephpassant/DissertationRedo/Informer2020/models/base_eval_jpformer_config.json"
    config_save_path = os.path.join(save_dir, "config.json")

    if validation_metrics_df is None or validation_metrics_df.empty:
        print("ERROR: validation_metrics_df is empty. Skipping save.")
    else:
        validation_metrics_df.to_csv(os.path.join(save_dir, "validation_metrics.csv"), index=False)
        print("Saved validation_metrics.csv")

    if detailed_df is None or detailed_df.empty:
        print("ERROR: actual_predicted_dy_df is empty. Skipping save.")
    else:
        detailed_df.to_csv(os.path.join(save_dir, "actual_and_predicted_values.csv"), index=False)
        print("Saved actual_and_predicted_values.json")

    if os.path.exists(config_path):
        shutil.copy(config_path, config_save_path)
        print(f"Copied config file to {config_save_path}")
    else:
        print(f"ERROR: Config file {config_path} not found. Skipping copy.")
    
    # Save CG-EGA statistics
    overall_cg_ega_df = stats_tables["overall"]
    overall_cg_ega_df.to_csv(os.path.join(save_dir, "overall_cg_ega.csv"))
    print("Saved overall CG-EGA statistics")

    # Save timepoint statistics
    for timepoint, df in stats_tables["timepoints"].items():
        df.to_csv(os.path.join(save_dir, f"cg_ega_stats_{timepoint}.csv"))
        print(f"Saved {timepoint} CG-EGA statistics")

    print("All CG-EGA statistics saved successfully.")
    
    # Save trained model
    model_filename = f"{model_name}_{model_type}_{test_rmse:.4f}_MAE_{test_mae:.4f}.pth"
    model_path = os.path.join(save_dir, model_filename)

    if model.state_dict():
        torch.save(model.state_dict(), model_path)
        print(f"Saved model at {model_path}")
    else:
        print("ERROR: model.state_dict() is empty. Model not saved.")

    print(f"Model and results saved to {save_dir}/")

