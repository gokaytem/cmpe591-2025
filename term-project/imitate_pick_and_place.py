import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
import time
import argparse
import re

# --- Hyperparameters ---
IMG_SIZE = 256          # Size (height and width) to which input images will be resized (in pixels)
BATCH_SIZE = 16         # Number of samples processed before the model is updated (mini-batch size)
NUM_EPOCHS = 200        # Number of times the entire training dataset will be passed through the model
LEARNING_RATE = 1e-4    # Step size for updating model weights during training
NUM_WORKERS = 0         # Number of subprocesses to use for data loading (0 means data loading is done in the main process)
PIN_MEMORY = True       # If True, DataLoader will copy Tensors into CUDA pinned memory (improves GPU transfer speed)
WEIGHT_DECAY = 1e-4     # Regularization parameter to prevent overfitting by penalizing large weights
PATIENCE = 20           # Number of epochs to wait for improvement before early stopping (prevents overtraining)

CHECKPOINT_DIR = "train"            # Directory where model checkpoints will be saved during training
MODEL_PATH = "pickplace_model.pt"   # File path for saving the final trained model
DEFAULT_IMG_DIR = "data/train_pick-and-place/A"                 # Default directory containing training images
DEFAULT_LABEL_FILE = "data/train_pick-and-place/A/labels.txt"   # Default file containing labels for training images

class PickPlaceDataset(Dataset):
    """
    PyTorch Dataset for pick-and-place regression tasks.

    Each sample consists of an RGB image and a label containing the pick (x, y) and place (x, y) coordinates.
    The dataset reads image filenames and corresponding pick/place positions from a label file.
    Optionally, images can be preloaded into memory for faster access.

    Args:
        img_dir (str): Directory containing the images.
        label_file (str): Path to the label file. Each line should contain the image filename and pick/place positions.
        preload (bool): If True, preload all images into memory at initialization.

    Example label file line:
        image_001.png [0.12, 0.34, 1.02] [0.56, 0.78, 1.02]
    """
    def __init__(self, img_dir, label_file, preload=False):
        self.img_dir = img_dir
        self.samples = []
        self.preload = preload
        with open(label_file, "r") as f:
            for line in f:
                # Use regex to extract two lists from the line
                matches = re.findall(r'\[([^\]]+)\]', line)
                if len(matches) < 2:
                    raise ValueError(f"Could not parse pick/place positions from line: {line}")
                pick_pos = [float(x.strip()) for x in matches[0].split(',')]
                place_pos = [float(x.strip()) for x in matches[1].split(',')]
                # Only use x, y for both pick and place
                self.samples.append((line.split()[0], [pick_pos[0], pick_pos[1]], [place_pos[0], place_pos[1]]))
        if self.preload:
            from typing import Optional
            self.images: list[Optional[torch.Tensor]] = [None] * len(self.samples)
            from concurrent.futures import ThreadPoolExecutor
            def load_img(idx_img):
                idx, (img_name, _, _) = idx_img
                img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img).astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2,0,1)
                return idx, img
            with ThreadPoolExecutor() as executor:
                for idx, img in executor.map(load_img, enumerate(self.samples)):
                    self.images[idx] = img

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get the (image, label) pair at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            img (torch.Tensor): Image tensor of shape (3, IMG_SIZE, IMG_SIZE), normalized to [0, 1].
            label (torch.Tensor): Tensor of shape (4,) containing [pick_x, pick_y, place_x, place_y].
        """
        img_name, pick_xy, place_xy = self.samples[idx]
        if self.preload:
            img = self.images[idx]
        else:
            img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2,0,1)
        label = np.array(pick_xy + place_xy, dtype=np.float32)  # shape (4,)
        label = torch.from_numpy(label)
        return img, label

class PickPlaceCNN(nn.Module):
    """
    Convolutional Neural Network for pick-and-place regression.

    This model takes an RGB image as input and predicts four values:
    [pick_x, pick_y, place_x, place_y], representing the (x, y) coordinates
    for pick and place positions.

    Architecture:
        - 3 convolutional layers with ReLU activations and max pooling
        - Flatten layer
        - Two fully connected (linear) layers with ReLU and dropout
        - Final linear layer outputs 4 regression values

    Input:
        x (torch.Tensor): Batch of images of shape (batch_size, 3, IMG_SIZE, IMG_SIZE)

    Output:
        torch.Tensor: Batch of predictions of shape (batch_size, 4)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.0),
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, 4)  # 4 outputs: pick(xy), place(xy)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, IMG_SIZE, IMG_SIZE)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 4)
        """
        return self.net(x)

def train_pickplace(
    model,
    train_loader,
    val_loader=None,
    epochs=NUM_EPOCHS,
    save_path=None,
    checkpoint_dir=CHECKPOINT_DIR
):
    """
    Trains a PyTorch model for a pick-and-place task using mean squared error loss and Adam optimizer.
    Supports checkpointing, early stopping, and learning curve plotting.
    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset. If None, validation is skipped. Default is None.
        epochs (int, optional): Number of training epochs. Default is NUM_EPOCHS.
        save_path (str, optional): Path to save the final trained model. If None, the model is not saved at the end. Default is None.
        checkpoint_dir (str, optional): Directory to save checkpoints and learning curves. Default is CHECKPOINT_DIR.
    Returns:
        None
    Side Effects:
        - Saves model checkpoints every 10 epochs in `checkpoint_dir`.
        - Saves the best model (lowest validation loss) as 'best_model.pt' in `checkpoint_dir`.
        - Saves learning curve data as 'learning_curve.npz' in `checkpoint_dir`.
        - Plots and saves loss and learning rate curves as 'loss_curve.png' and 'lr_curve.png' in `checkpoint_dir`.
        - Prints training and validation loss per epoch, and early stopping message if triggered.
    Notes:
        - Resumes training from the latest checkpoint in `checkpoint_dir` if available.
        - Uses early stopping based on validation loss with patience defined by PATIENCE.
        - Assumes global constants NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, and CHECKPOINT_DIR are defined.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.train()

    train_losses = []
    val_losses = []
    epoch_losses = []
    epoch_lrs = []

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "pickplace_model_*.pt")))
    start_epoch = 0
    if checkpoint_files:
        latest_ckpt = max(checkpoint_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
        print(f"Resuming from checkpoint: {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        start_epoch = int(os.path.splitext(os.path.basename(latest_ckpt))[0].split("_")[-1])

    best_val_loss = float('inf')
    patience = PATIENCE
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        total_loss = 0
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        epoch_losses.append(avg_loss)
        epoch_lrs.append(optimizer.param_groups[0]['lr'])
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f} sec")

        # Validation loss
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    preds = model(imgs)
                    loss = loss_fn(preds, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            print(f"  Validation Loss: {val_loss:.4f}")
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                    break
        # Save learning curve data after each epoch
        np.savez(os.path.join(checkpoint_dir, "learning_curve.npz"), losses=np.array(epoch_losses), lrs=np.array(epoch_lrs))
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"pickplace_model_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Plot learning curve at the end
    plt.figure()
    plt.plot(np.arange(1, len(epoch_losses)+1), epoch_losses, label="Training Loss")
    if val_losses:
        plt.plot(np.arange(1, len(val_losses)+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(np.arange(1, len(epoch_lrs)+1), epoch_lrs, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir, "lr_curve.png"))
    plt.close()

def predict_pickplace(model, img_path):
    """
    Predicts pick and place positions from an input image using a given model.

    Args:
        model (torch.nn.Module): The trained PyTorch model used for prediction.
        img_path (str): Path to the input image file.

    Returns:
        tuple:
            pick_pos (list of float): The predicted (x, y, z) coordinates for the pick position, 
                                      where z is set to 1.02.
            place_pos (list of float): The predicted (x, y, z) coordinates for the place position, 
                                       where z is set to 1.02.

    Notes:
        - The input image is resized to (256, 256) and normalized before prediction.
        - The function automatically selects CUDA if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(img_path).convert("RGB").resize((256,256))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        model = model.to(device)
        preds = model(img)
        preds = preds.squeeze(0).cpu().numpy()
        pick_xy = preds[:2].tolist()
        place_xy = preds[2:].tolist()
        pick_pos = pick_xy + [1.02]
        place_pos = place_xy + [1.02]
    return pick_pos, place_pos

if __name__ == "__main__":
    """
    Main entry point for training or testing the pick-and-place model.
    Usage:
        python imitate_pick_and_place.py --mode train --img_dir <image_dir> --label_file <label_file>
        python imitate_pick_and_place.py --mode test --model_path <model.pt> --test_img <image.png>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--img_dir", default=DEFAULT_IMG_DIR)
    parser.add_argument("--label_file", default=DEFAULT_LABEL_FILE)
    parser.add_argument("--test_img", default=None)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--pin_memory", type=bool, default=PIN_MEMORY)
    args = parser.parse_args()

    if args.mode == "train":
        dataset = PickPlaceDataset(args.img_dir, args.label_file, preload=True)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        # Split dataset into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        model = PickPlaceCNN()
        train_pickplace(
            model,
            train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            save_path=args.model_path,
            checkpoint_dir=args.checkpoint_dir
        )
    elif args.mode == "test":
        assert args.test_img is not None, "Please provide --test_img"
        model = PickPlaceCNN()
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        model.eval()
        pick_pos, place_pos = predict_pickplace(model, args.test_img)
        print(f"Predicted pick position: {pick_pos}")
        print(f"Predicted place position: {place_pos}")
