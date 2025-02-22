import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Formula to calculate output size:
        # Output size = ((Input size - Kernel size + 2 * Padding) / Stride) + 1
        self.image_conv1 = nn.Conv2d(3, 2, kernel_size=4, stride=2, padding=1)
        # For conv1 with input size 128x128: Output size = ((128 - 4 + 2 * 1) / 2) + 1 = 64
        self.image_conv2 = nn.Conv2d(2, 1, kernel_size=4, stride=4, padding=0)
        # For conv2: Output size = ((64 - 4 + 2 * 0) / 4) + 1 = 16
        
        self.action_fc1 = nn.Linear(1, 32)

        self.combined_fc1 = nn.Linear(1 * 16 * 16 + 32, 128)  # Verify the input size calculation
        self.combined_fc2 = nn.Linear(128, 2)

    def forward(self, img, action):
        img = F.relu(self.image_conv1(img))
        img = F.relu(self.image_conv2(img))
        img = torch.flatten(img, start_dim=1)  # Flatten the tensor

        action = F.relu(self.action_fc1(action))
        
        combined = torch.cat((img, action), dim=1)
        combined = F.relu(self.combined_fc1(combined))
        combined = self.combined_fc2(combined)
        return combined

def load_data(idx):
    positions = torch.load(f"./hw1_sample/positions_{idx}.pt")
    actions = torch.load(f"./hw1_sample/actions_{idx}.pt")
    imgs_before = torch.load(f"./hw1_sample/imgsBefore_{idx}.pt")
    return positions, actions, imgs_before

def load_test_data(idx):
    positions = torch.load(f"./hw1_test/positions_{idx}.pt")
    actions = torch.load(f"./hw1_test/actions_{idx}.pt")
    imgs_before = torch.load(f"./hw1_test/imgsBefore_{idx}.pt")
    return positions, actions, imgs_before

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_imgs, batch_actions, batch_positions in dataloader:
            outputs = model(batch_imgs, batch_actions)
            loss = criterion(outputs, batch_positions)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def train():
    # Load data for all indices
    all_positions = []
    all_actions = []
    all_imgs_before = []
    for i in range(8):
        positions, actions, imgs_before = load_data(i)
        all_positions.append(positions)
        all_actions.append(actions)
        all_imgs_before.append(imgs_before)

    # Concatenate all data
    positions = torch.cat(all_positions)
    actions = torch.cat(all_actions)
    imgs_before = torch.cat(all_imgs_before)

    # Convert actions and images to float
    actions = actions.view(-1, 1).float()
    imgs_before = imgs_before.float() / 255.0  # Normalize the images to [0, 1]

    # Create a dataset and dataloader
    dataset = TensorDataset(imgs_before, actions, positions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNN()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    loss_values = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started.")
        epoch_loss = 0
        for batch_idx, (batch_imgs, batch_actions, batch_positions) in enumerate(dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_imgs, batch_actions)

            # Compute loss
            loss = criterion(outputs, batch_positions)

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
    torch.save(model.state_dict(), "hw1_2.pt")

    # Plot the loss values and save the plot
    plt.plot(range(1, num_epochs + 1), loss_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence of CNN')
    plt.ylim(0, 0.05)  # Set y-axis limits between 0 and 0.05
    plt.savefig('hw1_2.png')
    plt.close()

def test():
    # Load test data for all indices
    all_positions = []
    all_actions = []
    all_imgs_before = []
    for i in range(8):
        positions, actions, imgs_before = load_test_data(i)
        all_positions.append(positions)
        all_actions.append(actions)
        all_imgs_before.append(imgs_before)

    # Concatenate all test data
    positions = torch.cat(all_positions)
    actions = torch.cat(all_actions)
    imgs_before = torch.cat(all_imgs_before)

    # Convert actions and images to float
    actions = actions.view(-1, 1).float()
    imgs_before = imgs_before.float() / 255.0  # Normalize the images to [0, 1]

    # Create a dataset and dataloader for evaluation
    dataset = TensorDataset(imgs_before, actions, positions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load and evaluate the CNN model
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load('hw1_2.pt'))
    cnn_loss = evaluate_model(cnn_model, dataloader)
    print(f"Average Evaluation Loss for CNN: {cnn_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test the CNN model")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode to run: 'train' or 'test'")
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()