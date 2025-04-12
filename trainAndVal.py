import os
import time
import torch
import numpy as np
import nibabel as nib
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
from UnetEncoder import UNet3DAutoencoder  # Ensure this module is correctly implemented

# Define the dataset path
input_dir = "/projectnb/nsa-aphasia/gsummers/04-3DUNet-CorticalThickness/data/HCP/T1w/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and preprocess an MRI file
def load_mri(file_path, original_shape=(182, 218, 182), scale_factor=4):
    img = nib.load(file_path).get_fdata()
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Min-max normalization
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    target_shape = tuple(s // scale_factor for s in original_shape)
    img = F.interpolate(img.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=False).squeeze(0)
    return img

# Custom dataset class
class MRI_Dataset(Dataset):
    def __init__(self, file_list, original_shape=(182, 218, 182), scale_factor=4):
        self.file_list = file_list
        self.original_shape = original_shape
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = load_mri(self.file_list[idx], self.original_shape, self.scale_factor)
        return img

# Collect all T1-weighted MRI files
t1w_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith((".nii", ".nii.gz"))]
dataset = MRI_Dataset(t1w_files)

# Define 10-fold cross-validation
k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Latent sizes and pooling layers to test
latent_sizes = [128]
pooling_layers = [4]
epochs = 4
lr = 1e-3
early_stop_patience = 3  # Stop if loss doesn't improve for 3 consecutive epochs

# Function to train the model
def train_autoencoder(model, train_loader, val_loader, latent_dim, pooling_layers, epochs=4, lr=1e-3, device="cuda"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    losses = []
    val_losses = []
    best_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Latent {latent_dim}, Pooling {pooling_layers} - Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_latent_{latent_dim}_pooling_{pooling_layers}.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} for latent {latent_dim}, pooling {pooling_layers}")
                break

    training_time = time.time() - start_time
    return losses, val_losses, training_time

# Dictionary to store results
results = {}

# Perform k-fold cross-validation
for latent_dim in latent_sizes:
    for pooling_layer in pooling_layers:
        fold_train_losses = []
        fold_val_losses = []
        fold_training_times = []

        print(f"\nTraining model with latent size: {latent_dim} and pooling layers: {pooling_layer}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"\nFold {fold+1}/{k_folds}")

            # Create training and validation datasets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # Create dataloaders
            train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

            # Initialize model
            model = UNet3DAutoencoder(latent_dim=latent_dim, pooling_layers=pooling_layer)

            # Train model
            train_losses, val_losses, training_time = train_autoencoder(model, train_loader, val_loader, 
                                                                        latent_dim, pooling_layer, 
                                                                        epochs, lr, device)
            fold_train_losses.append(train_losses)
            fold_val_losses.append(val_losses)
            fold_training_times.append(training_time)

        # Compute average performance across folds
        avg_train_losses = np.mean(fold_train_losses, axis=0)
        avg_val_losses = np.mean(fold_val_losses, axis=0)
        avg_training_time = np.mean(fold_training_times)

        results[(latent_dim, pooling_layer)] = {
            "train_losses": avg_train_losses,
            "val_losses": avg_val_losses,
            "time": avg_training_time
        }

# Plot the results
plt.figure(figsize=(10, 6))
for (latent_dim, pooling_layer), data in results.items():
    plt.plot(range(1, len(data["val_losses"]) + 1), data["val_losses"], label=f"Latent {latent_dim}, Pooling {pooling_layer}")

plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Across Latent Sizes and Pooling Layers (10-Fold CV)")
plt.legend()
plt.savefig("latent_size_pooling_comparison_cv.png")
plt.show()

# Print average training times
for (latent_dim, pooling_layer), data in results.items():
    print(f"Latent {latent_dim}, Pooling {pooling_layer}: Average Training Time = {data['time']:.2f} sec")

