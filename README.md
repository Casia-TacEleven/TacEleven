# Code and Homepage for TacEleven
Homepage: https://casia-taceleven.github.io/TacEleven/
This repository contains the code for TacEleven: generator training and critic pipeline.

![TacEleven Overview](static/images/overflow.jpg)

## Repository Overview

This repository provides the implementation of TacEleven, a framework designed for advanced generator training and a robust critic pipeline. It includes the following components:

### Features
- **Generator Training**: Tools and scripts to train generative models with high efficiency and accuracy.
- **Critic Pipeline**: A modular pipeline for evaluating and improving the performance of generative models.

### Getting Started
To get started with TacEleven, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Casia-TacEleven/TacEleven.git

# Navigate to the project directory
cd TacEleven

# Install the required dependencies
pip install -r requirements.txt

# Train the generator model
bash generator/launch_tmux_sample_point_variable.sh

# Run the critic pipeline
python critic/main_delicated.py
```
