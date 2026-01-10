# **DWMR: Discrete World Models via Regularization**

This repository contains the official implementation of **Discrete World Models via Regularization (DWMR)**.  
DWMR is a reconstruction-free and contrastive-free method for learning unsupervised Boolean world models. By combining a Joint-Embedding Predictive Architecture with specialized regularizers, DWMR learns informative Boolean latent representations and transitions.  
DWMR optimizes a composite objective function $\\mathcal{L}\_{DWMR}$:  
$$\\mathcal{L}\_{DWMR} \= \\mathcal{L}\_{pred} \+ \\lambda\_{var}\\mathcal{L}\_{var} \+ \\lambda\_{cor}\\mathcal{L}\_{cor} \+ \\lambda\_{cos}\\mathcal{L}\_{cos} \+ \\lambda\_{loc}\\mathcal{L}\_{loc}$$

* $\\mathcal{L}\_{pred}$ ensures the model anticipates future latent states accurately.  
* $\\mathcal{L}\_{var}$ prevents informational collapse by ensuring each bit maintains high variance.  
* $\\mathcal{L}\_{cor}$ prevents informational collapse by minimizing redundancy between bits.  
* $\\mathcal{L}\_{cos}$ reduces third-order dependencies between latent bits for better disentanglement.  
* $\\mathcal{L}\_{loc}$ penalizes large Hamming distances between consecutive states.

## **Project Structure**

The codebase is organized as follows:

* `architecture.py`: Implements the world model, with options for the different architectural variants and features.  
* `encoder_decoder.py`: Defines the CNN-based encoders and decoders.  
* `loss_functions.py`: Implements the DWMR loss suite (pred, var, cor, cos, loc, rec, KL).  
* `experiment.py`: Main entry point for training, evaluation, and hyperparameter optimization.  
* `dataset.py`: Data generation utilities for both IceSlider and MNIST 8-puzzle.  
* `utils.py`: Helper functions for logging, probing, and performance evaluation.

## **Installation & Setup**

### **1\. Environment Setup**

We recommend using **Conda** to manage a clean environment. DWMR requires Python 3.9 or higher.  
```
conda create dwmr python=3.11  
conda activate dwmr
```

### **2\. Install Core Dependencies**

Install the primary deep learning and optimization libraries. For PyTorch, please refer to the [official instructions](https://pytorch.org/get-started/locally/) to match your CUDA version.  
```
# Example for CUDA 11.8  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optimization and Experiment Tracking  
pip install wandb optuna tqdm noise pillow scipy scikit-learn matplotlib
```

### **3\. External Library: Puzzlegen**

The **IceSlider** environment requires the puzzlegen library. Install it from the source:  
```
git clone https://github.com/martius-lab/puzzlegen.git
cd puzzlegen  
pip install -e .  
cd ..
```

### **4\. Tracking and Logging**

This project uses **Weights & Biases** for experiment logging and hyperparameter sweeps. To enable this, log in to your account:  
```
wandb login
```

## **Quickstart**

All experiments are run from `experiment.py`.

```
# 8-puzzle clean
python experiment.py --experimentname dwmr-8puzzle-clean --extrastep --scheduling

# 8-puzzle noisy
python experiment.py --experimentname dwmr-8puzzle-noisy --noisetype gaussian --extrastep --scheduling

# IceSlider clean
python experiment.py --experimentname dwmr-iceslider-clean --iceslider --extrastep --scheduling --nbits 192 --epochs 20 --interval 4
```

By default, the script will:

- Generate datasets into `data/` if they are missing.
- Run **Optuna hyperparameter tuning** (`--trials` controls the budget).
- Train with the best found hyperparameters and compute probe metrics.
- Save results into `data/res.json`.

--- 

Key CLI flags (see `experiment.py` for the full list):

- **Dataset**
  - `--iceslider`: use IceSlider instead of the MNIST 8-puzzle
  - `--noisetype {none,gaussian}` and `--noisestrength <float>`

- **Training**

  - `--epochs <int>` (default: 40)
  - `--batchsize <int>` (default: 256)
  - `--repetitions <int>`: number repeated runs
  - `--interval <int>`: how often (in epochs) to train probes and log metrics
  - `--scheduling`: apply exponential schedules to hyperparameters
  - `--extrastep`: enable the 2-steps per minibatch training scheme

- **Model variants / ablations**

  - `--decoder`: add a reconstruction decoder (DWMR+AE if regularizers are on)
  - `--variational`: use relaxed sampling + KL regularization (β-VAE-like)
  - `--noreg`: disable DWMR regularizers (variance/cov/third/locality)
  - `--deepcubeai`: enable a DeepCubeAI-style baseline configuration
  - `--ablation`: run systematic ablations (uses stored hyperparameters)

- **Hyperparameter tuning**

  - `--trials <int>`: Optuna trial budget
  - `--nohypertuning`: skip Optuna and load hyperparameters from `data/res.json`

