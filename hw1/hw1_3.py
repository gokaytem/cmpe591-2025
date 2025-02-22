import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        # Encoder for images
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU()
        )
        # Encoder for actions
        self.action_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128 + 32, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.Sigmoid()  # Use sigmoid to get output in range [0, 1]
        )

    def forward(self, img, action):
        # Encoder output shape: [batch, 128, 8, 8]
        img_encoded = self.encoder(img)
        
        # Action encoding shape: [batch, 32] -> reshape/expand -> [batch, 32, 8, 8]
        action_encoded = self.action_encoder(action).unsqueeze(2).unsqueeze(3)
        action_encoded = action_encoded.expand(-1, -1, img_encoded.size(2), img_encoded.size(3))

        # Concatenate along channel dimension -> [batch, 128+32, 8, 8]
        combined = torch.cat((img_encoded, action_encoded), dim=1)
        
        # Pass the 4D tensor into the convolutional decoder
        output = self.decoder(combined)
        return output

def load_data(idx):
    imgs_before = torch.load(f"./hw1_sample/imgsBefore_{idx}.pt")
    imgs_after = torch.load(f"./hw1_sample/imgsAfter_{idx}.pt")
    actions = torch.load(f"./hw1_sample/actions_{idx}.pt")
    return imgs_before, imgs_after, actions

def load_test_data(idx):
    imgs_before = torch.load(f"./hw1_test/imgsBefore_{idx}.pt")
    imgs_after = torch.load(f"./hw1_test/imgsAfter_{idx}.pt")
    actions = torch.load(f"./hw1_test/actions_{idx}.pt")
    return imgs_before, imgs_after, actions

def train():
    # Load data for all indices
    all_imgs_before = []
    all_imgs_after = []
    all_actions = []
    for i in range(8):
        imgs_before, imgs_after, actions = load_data(i)
        all_imgs_before.append(imgs_before)
        all_imgs_after.append(imgs_after)
        all_actions.append(actions)

    # Concatenate all data
    imgs_before = torch.cat(all_imgs_before)
    imgs_after = torch.cat(all_imgs_after)
    actions = torch.cat(all_actions)

    # Normalize the images to [0, 1]
    imgs_before = imgs_before.float() / 255.0
    imgs_after = imgs_after.float() / 255.0
    actions = actions.view(-1, 1).float()

    # Create a dataset and dataloader
    dataset = TensorDataset(imgs_before, imgs_after, actions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Reconstructor()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    loss_values = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started.")
        epoch_loss = 0
        for batch_idx, (batch_imgs_before, batch_imgs_after, batch_actions) in enumerate(dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_imgs_before, batch_actions)

            # Compute loss
            loss = criterion(outputs, batch_imgs_after)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        average_epoch_loss = epoch_loss / len(dataloader)
        loss_values.append(average_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished with Loss: {average_epoch_loss:.4f}")

    print("Finished Training")
    
    # Save the model
    torch.save(model.state_dict(), "hw1_3.pt")

    # Plot the loss values and save the plot
    plt.plot(range(1, num_epochs + 1), loss_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence of Reconstructor')
    plt.ylim(0, 0.01)  # Set y-axis limits between 0 and 0.01
    plt.savefig('hw1_3.png')
    plt.close()

def test():
    # Load test data for all indices
    all_imgs_before = []
    all_imgs_after = []
    all_actions = []
    for i in range(8):
        imgs_before, imgs_after, actions = load_test_data(i)
        all_imgs_before.append(imgs_before)
        all_imgs_after.append(imgs_after)
        all_actions.append(actions)

    # Concatenate all test data
    imgs_before = torch.cat(all_imgs_before)
    imgs_after = torch.cat(all_imgs_after)
    actions = torch.cat(all_actions)

    # Normalize the images to [0, 1]
    imgs_before = imgs_before.float() / 255.0
    imgs_after = imgs_after.float() / 255.0
    actions = actions.view(-1, 1).float()

    # Create a dataset and dataloader for evaluation
    dataset = TensorDataset(imgs_before, imgs_after, actions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load and evaluate the Reconstructor model
    reconstructor_model = Reconstructor()
    reconstructor_model.load_state_dict(torch.load('hw1_3.pt'))
    reconstructor_model.eval()
    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (batch_imgs_before, batch_imgs_after, batch_actions) in enumerate(dataloader):
            outputs = reconstructor_model(batch_imgs_before, batch_actions)
            loss = criterion(outputs, batch_imgs_after)
            total_loss += loss.item()

            # Save the resulting images next to imgsAfter
            for i in range(outputs.size(0)):
                combined = torch.cat((batch_imgs_after[i], outputs[i]), dim=2)  # Concatenate images side-by-side
                save_image(combined, f"./hw1_3_output/combined_{batch_idx * dataloader.batch_size + i}.png")

    average_loss = total_loss / len(dataloader)
    print(f"Average Evaluation Loss for Reconstructor: {average_loss:.4f}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test the Reconstructor model")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode to run: 'train' or 'test'")
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()