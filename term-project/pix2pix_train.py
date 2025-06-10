import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import argparse
import matplotlib.pyplot as plt

# --- Hyperparameters ---
GEN_DROPOUT = 0.0           # Dropout rate for generator (0.0-0.5, usually 0.0 or 0.2)
DISC_DROPOUT = 0.0          # Dropout rate for discriminator (0.0-0.5, usually 0.0)
L1_WEIGHT = 100             # Weight for L1 loss (10-200, typical: 100)
BATCH_SIZE = 4              # Batch size for training (1-16, typical: 1, 4, or 8)
EPOCHS = 100                # Number of training epochs (20-200+)
LR = 2e-4                   # Learning rate for Adam optimizer (1e-5 to 2e-4, typical: 2e-4)
BETAS = (0.5, 0.999)        # Adam optimizer betas (default for GANs: (0.5, 0.999))
IMG_SIZE = 256              # Image size (64, 128, 256, 512; typical: 256)
LABEL_SMOOTH_REAL = 1.0     # Real label for GAN loss (1.0 for no smoothing, 0.8-1.0 for smoothing)
LABEL_SMOOTH_FAKE = 0.0     # Fake label for GAN loss (0.0 for no smoothing, 0.0-0.2 for smoothing)
SAVE_EVERY = 5              # Save checkpoint every N epochs

TRAIN_DIR_A = "data/train_pix2pix/B"   # Directory for input images (domain B)
TRAIN_DIR_B = "data/train_pix2pix/A"   # Directory for target images (domain A)
CHECKPOINT_DIR = "pix2pix_checkpoints" # Directory to save model checkpoints
INFER_INPUT_DIR = "data/stepwise/B"    # Directory for inference input images
INFER_OUTPUT_DIR = "data/stepwise/B2A" # Directory for inference output images

class Generator(nn.Module):
    """
    Generator neural network for image-to-image translation tasks.

    This class defines a simple convolutional generator model using PyTorch's nn.Module.
    The architecture consists of a sequence of convolutional layers with ReLU activations,
    dropout for regularization, and a final Tanh activation to output images in the range [-1, 1].

    Attributes:
        model (nn.Sequential): The sequential container of convolutional, activation, and dropout layers.

    Methods:
        forward(x):
            Performs a forward pass of the input tensor x through the generator network.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 3, H, W), representing the generated image.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(),
            nn.Dropout(GEN_DROPOUT),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.Dropout(GEN_DROPOUT),
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
            nn.Dropout(GEN_DROPOUT),
            nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """
    Discriminator neural network for image-to-image translation tasks.

    This class defines a PatchGAN-style discriminator model using PyTorch's nn.Module.
    The discriminator takes as input the concatenation of the input image and either the real or generated target image,
    and outputs a probability map indicating the realism of each patch.

    Attributes:
        model (nn.Sequential): The sequential container of convolutional, activation, dropout, and normalization layers.

    Methods:
        forward(x, y):
            Performs a forward pass of the concatenated input and target/generated images through the discriminator network.

    Args:
        x (torch.Tensor): Input tensor (e.g., source image) of shape (batch_size, 3, H, W).
        y (torch.Tensor): Target or generated tensor of shape (batch_size, 3, H, W).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1, H', W'), representing the probability map for each patch.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Dropout(DISC_DROPOUT),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Dropout(DISC_DROPOUT),
            nn.Conv2d(128, 1, 4, 1, 1), nn.Sigmoid()
        )
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

class PairedDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for paired image-to-image translation tasks (e.g., pix2pix).

    This dataset matches images from two directories (A and B) based on a shared numeric identifier
    in their filenames (e.g., 'image_00001.png' in both directories will be paired together).
    Only files with matching numeric parts in both directories are included.

    Args:
        dir_A (str): Path to the directory containing images from domain A.
        dir_B (str): Path to the directory containing images from domain B.
        size (int, optional): The size to which images will be resized (square). Defaults to IMG_SIZE.

    Attributes:
        files_A (List[str]): Sorted list of filenames from dir_A with matching pairs in dir_B.
        files_B (List[str]): Sorted list of filenames from dir_B with matching pairs in dir_A.
        dir_A (str): Directory path for domain A images.
        dir_B (str): Directory path for domain B images.
        transform (torchvision.transforms.Compose): Transformations applied to both images.

    Methods:
        __len__(): Returns the number of paired samples.
        __getitem__(idx): Returns a tuple of (image_A, image_B) as tensors, both normalized to [-1, 1].
    """
    def __init__(self, dir_A, dir_B, size=IMG_SIZE):
        # Match files by numeric part (e.g., 00001)
        def get_num(fname):
            # Extracts the numeric part before .png/.jpg
            base = os.path.splitext(fname)[0]
            return base.split('_')[-1]
        files_A = {get_num(f): f for f in os.listdir(dir_A) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        files_B = {get_num(f): f for f in os.listdir(dir_B) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        common_keys = sorted(set(files_A.keys()) & set(files_B.keys()))
        self.files_A = [files_A[k] for k in common_keys]
        self.files_B = [files_B[k] for k in common_keys]
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self):
        return len(self.files_A)
    def __getitem__(self, idx):
        img_A = Image.open(os.path.join(self.dir_A, self.files_A[idx])).convert("RGB")
        img_B = Image.open(os.path.join(self.dir_B, self.files_B[idx])).convert("RGB")
        return self.transform(img_A), self.transform(img_B)

def train_pix2pix(
    dir_A=TRAIN_DIR_A, dir_B=TRAIN_DIR_B,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, save_every=SAVE_EVERY, device="cuda"
):
    """
    Trains a Pix2Pix model for paired image-to-image translation.
    This function sets up the dataset, initializes the generator and discriminator networks,
    handles checkpointing for resuming training, and performs adversarial training using
    both GAN and L1 losses. It periodically saves model checkpoints and sample outputs,
    and plots the generator and discriminator losses after training.
    Args:
        dir_A (str): Directory containing input images (domain A).
        dir_B (str): Directory containing target images (domain B).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizers.
        save_every (int): Frequency (in epochs) to save checkpoints and sample outputs.
        device (str): Device to use for training ('cuda' or 'cpu').
    Side Effects:
        - Saves model checkpoints and sample images to CHECKPOINT_DIR.
        - Plots and saves a loss curve after training.
    Raises:
        RuntimeError: If the discriminator output shape cannot be determined (e.g., empty dataset).
    Returns:
        None
    """
    dataset = PairedDataset(dir_A, dir_B)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    G = Generator().to(device)
    D = Discriminator().to(device)
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=BETAS)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=BETAS)

    # Resume from latest checkpoint if available
    start_epoch = 0
    checkpoint_dir = CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Find latest checkpoint
    gen_ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("generator_epoch") and f.endswith(".pth")]
    disc_ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("discriminator_epoch") and f.endswith(".pth")]
    if gen_ckpts and disc_ckpts:
        # Get latest epoch number
        latest_epoch = max(int(f.split("_epoch")[1].split(".")[0]) for f in gen_ckpts)
        gen_path = os.path.join(checkpoint_dir, f"generator_epoch{latest_epoch}.pth")
        disc_path = os.path.join(checkpoint_dir, f"discriminator_epoch{latest_epoch}.pth")
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            print(f"Resuming from checkpoint at epoch {latest_epoch}")
            G.load_state_dict(torch.load(gen_path, map_location=device))
            D.load_state_dict(torch.load(disc_path, map_location=device))
            start_epoch = latest_epoch

    # Get output shape of discriminator for dynamic valid/fake tensor size
    disc_out_shape = None
    with torch.no_grad():
        for real_A, real_B in loader:
            real_A, real_B = real_A.to(device), real_B.to(device)
            fake_B = G(real_A)
            pred_fake = D(real_A, fake_B)
            disc_out_shape = pred_fake.shape
            break
    if disc_out_shape is None:
        raise RuntimeError("Could not determine discriminator output shape. Check your dataset.")

    gen_losses = []
    disc_losses = []
    for epoch in range(start_epoch, epochs):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        n_batches = 0
        for i, (real_A, real_B) in enumerate(loader):
            real_A, real_B = real_A.to(device), real_B.to(device)

            # Forward pass for generator
            fake_B = G(real_A)
            pred_fake = D(real_A, fake_B)
            valid = torch.ones_like(pred_fake, device=device) * LABEL_SMOOTH_REAL
            fake = torch.zeros_like(pred_fake, device=device) + LABEL_SMOOTH_FAKE
            
            # Train Generator
            optimizer_G.zero_grad()
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_L1 = criterion_L1(fake_B, real_B)
            loss_G = loss_GAN + L1_WEIGHT * loss_L1
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = D(real_A, real_B)
            valid_D = torch.ones_like(pred_real, device=device) * LABEL_SMOOTH_REAL
            loss_real = criterion_GAN(pred_real, valid_D)
            pred_fake = D(real_A, fake_B.detach())
            fake_D = torch.zeros_like(pred_fake, device=device) + LABEL_SMOOTH_FAKE
            loss_fake = criterion_GAN(pred_fake, fake_D)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            epoch_gen_loss += loss_G.item()
            epoch_disc_loss += loss_D.item()
            n_batches += 1

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(loader)}] "
                      f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} L1: {loss_L1.item():.4f}")

        # Average loss for the epoch
        if n_batches > 0:
            gen_losses.append(epoch_gen_loss / n_batches)
            disc_losses.append(epoch_disc_loss / n_batches)

        if (epoch + 1) % save_every == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(G.state_dict(), f"{CHECKPOINT_DIR}/generator_epoch{epoch+1}.pth")
            torch.save(D.state_dict(), f"{CHECKPOINT_DIR}/discriminator_epoch{epoch+1}.pth")
            # Save sample output: fromA, fromB, generated
            with torch.no_grad():
                sample_A, sample_B = next(iter(loader))
                sample_A = sample_A.to(device)
                sample_B = sample_B.to(device)
                sample_fake_B = G(sample_A)
                # Denormalize for visualization
                def denorm(x):
                    return (x * 0.5 + 0.5).clamp(0, 1)
                # Take first sample in batch for visualization
                img_A = denorm(sample_A[0].cpu())
                img_B = denorm(sample_B[0].cpu())
                img_fake_B = denorm(sample_fake_B[0].cpu())
                # Concatenate horizontally
                grid = torch.cat([img_A, img_B, img_fake_B], dim=2)  # along width
                save_image(grid, f"pix2pix_checkpoints/epoch{epoch+1}_A_B_fakeB.png")

    # Plot losses after training
    plt.figure()
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Pix2Pix Training Loss")
    plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_plot.png"))
    plt.show()

def get_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """
    Retrieves the latest generator checkpoint file from the specified directory.

    Args:
        checkpoint_dir (str): Path to the directory containing checkpoint files. Defaults to CHECKPOINT_DIR.

    Returns:
        tuple: A tuple containing the path to the latest generator checkpoint file (str) and the corresponding epoch number (int).

    Raises:
        FileNotFoundError: If no generator checkpoint files are found in the directory.
    """
    gen_ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("generator_epoch") and f.endswith(".pth")]
    if not gen_ckpts:
        raise FileNotFoundError("No generator checkpoints found.")
    latest_epoch = max(int(f.split("_epoch")[1].split(".")[0]) for f in gen_ckpts)
    gen_path = os.path.join(checkpoint_dir, f"generator_epoch{latest_epoch}.pth")
    return gen_path, latest_epoch

def inference_on_folder(input_dir=INFER_INPUT_DIR, output_dir=INFER_OUTPUT_DIR, device="cuda"):
    """
    Runs inference using the latest trained generator on all images in a folder.

    This function loads the latest generator checkpoint, applies the generator to each image
    in the specified input directory, and saves the generated images to the output directory.

    Args:
        input_dir (str): Directory containing input images for inference.
        output_dir (str): Directory to save generated output images.
        device (str): Device to run inference on ('cuda' or 'cpu').

    Side Effects:
        - Saves generated images to output_dir, preserving original filenames.
        - Prints progress for each saved image.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    gen_path, latest_epoch = get_latest_checkpoint()
    print(f"Using generator checkpoint: {gen_path}")
    G = Generator().to(device)
    G.load_state_dict(torch.load(gen_path, map_location=device))
    G.eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for fname in files:
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            fake_img = G(img_tensor)
        fake_img = fake_img.squeeze(0).cpu()
        fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)
        save_path = os.path.join(output_dir, fname)
        save_image(fake_img, save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    """
    Main entry point for training or inference.
    This script can be run in two modes:
    - 'train': Trains the Pix2Pix model on paired datasets.
    - 'infer': Runs inference on a folder of images using the latest trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train_pix2pix()
    elif args.mode == "infer":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inference_on_folder(device=device)
