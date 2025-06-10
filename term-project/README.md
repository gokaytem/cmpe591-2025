# CMPE591 Term Project: Imitation Learning for Pick-and-Place with Pix2Pix

## Overview

This project explores imitation learning for robotic pick-and-place tasks using deep learning and domain adaptation. Leveraging two visually and physically distinct MuJoCo environments, we train a convolutional neural network to predict pick and place positions directly from RGB images. To address the domain gap between environments, we employ a pix2pix model for image-to-image translation, enabling robust policy transfer. The repository includes environment setup, data sampling, model training, and evaluation scripts, providing a comprehensive framework for research in visual imitation learning and sim-to-sim transfer.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The goal of this project is to learn a policy that can imitate a pick-and-place task in a simulated environment. The project includes:

- Two MuJoCo environments (`A` and `B`) with different visual and physical properties.
- A shared environment builder for flexible scene creation.
- A PyTorch-based convolutional neural network that predicts pick and place positions from RGB images.
- Scripts for training and testing the imitation model.
- Utilities for stepwise control and evaluation of the learned policy.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/cmpe591-2025-term-project.git
    cd cmpe591-2025-term-project/term-project
    ```

2. **Install dependencies:**
    - Python 3.9+
    - [MuJoCo](https://mujoco.org/)
    - [dm_control](https://github.com/deepmind/dm_control)
    - [PyTorch](https://pytorch.org/)
    - numpy, matplotlib, pillow, scipy

    You can install Python dependencies with:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: You may need to install MuJoCo and dm_control manually according to their documentation.*

## Usage

### Sampling Training Data
To collect training data, run the data sampling script:
1.  Sample data for pix2pix training by setting the `env_version` to "both".
2.  Sample data for imitation learning by setting the `env_version` to "A".

```bash
python sample_data.py
```

- This will generate RGB images and corresponding label files in the specified directory.

### Training

To train the imitation model on pick-and-place demonstrations:

```bash
python imitate_pick_and_place.py
```

- The model will be trained for the environment A by default.
- Training checkpoints and loss curves will be saved in the `train/` directory.

To train the pix2pix model:

```bash
python pix2pix_train.py
```

- The model will be trained for the transformation of images from environments B to A by default.
- Training checkpoints and loss curves will be saved in the `pix2pix_checkpoints/` directory.

### Testing

To test the trained model via imitation learning on pick and place task:

```bash
python test_pickplace.py
```
- The model will be tested for the environment A by default.

To test the trained model via imitation learning on pick and place task with the images generated via pix2pix:

```bash
python test_pix2pix_pickplace.py
```
- The model will be tested for the transformation of images from environments B to A by default.

## Project Structure

```
term-project/
│
├── base_env_a.py               # Environment A definition
├── base_env_b.py               # Environment B definition
├── env_shared.py               # Shared environment utilities and logic
├── environment.py              # Environment factory
├── sample_env.py               # Samples the environment
├── imitate_pick_and_place.py   # Training and inference script for imitation learninig
├── pix2pix_train.py            # Training and inference script for pix2pix
├── test_pickplace.py           # Tests the imitation learning model
├── test_pix2pix_pickplace.py   # Tests the imitation learning model on generated images by pix2pix (sim2sim)
├── mujoco_menagerie/           # MuJoCo assets
├── wood.png, floor.png         # Texture images
├── README.md
└── requirements.txt
```