# Prostate MRI Classification

This repository contains code for classifying prostate MRI images using deep learning techniques.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/KevinMathewT/prostate-mri-classification.git
   cd prostate-mri-classification
   ```

2. Install dependencies using [Poetry](https://python-poetry.org/):
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Running the Project

### Training the Model
1. Create a configuration file or use an existing one from the `config/` directory. Example configurations may include hyperparameters, dataset paths, and training settings.
2. Run the training script with the chosen configuration file:
   ```bash
   accelerate launch -m train config/resnet50.yaml
   ```

   Replace `path/to/config.yaml` with the actual path to your configuration file.

3. The script will handle model initialization, training, validation, and logging based on the provided configuration.

### Validation
You can load and evaluate trained models using code snippets provided in `notebooks/` for further analysis and visualization.

## Repository Structure

- **`config/`**: Contains YAML configuration files for specifying training parameters.
- **`loader/`**: Code for loading and preprocessing MRI datasets.
- **`model/`**: Definitions of model architectures used for classification.
- **`notebooks/`**: Jupyter notebooks for exploration and evaluation of model performance.
- **`criterion.py`**: Implements custom loss functions for the training process.
- **`engine.py`**: Core training and evaluation logic, including epochs and batch handling.
- **`optimizer.py`**: Optimizer configurations for training.
- **`train.py`**: Main training script, which integrates the components and runs the training pipeline.
- **`utils.py`**: Utility functions for setup, initialization, and debugging.

Feel free to explore and modify the repository to suit your requirements!
