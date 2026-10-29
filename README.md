# Classifier PyTorch

A simple feed-forward neural network (FFNN) built with PyTorch to recognise handwritten digits, using MNIST's 60,000 training examples.

## Getting started

Find the latest stable pytorch versions [here](https://pytorch.org/get-started/locally/).

```bash
python3 -m venv .venv
source .venv/bin/activate

# Check CUDA Version
nvidia-smi

# Install dependencies and the local package
pip install -r requirements.txt

# Download the MNIST dataset
classifier data download

# View training examples
classifier data explore

# Predict one MNIST sample using random weights
classifier predict --random --sample 0

# Train a model and save its weights and settings
classifier train --epochs 5 --save ./model.pt

# Show sample 0 and its prediction after each training epoch
classifier train --epochs 5 --show-prediction

# Predict one MNIST sample using saved weights
classifier predict --load ./model.pt --sample 0

# Inspect checkpoint settings and tensors
classifier inspect ./model.pt
```

### Web interface

The optional frontend uses React, Vite, TypeScript and Tailwind. FastAPI serves an in-memory training session. Download MNIST using the CLI before starting.

```bash
# Install dependencies
source .venv/bin/activate
pip install -e ".[web]"
npm --prefix frontend ci

# Start the API server
python -m uvicorn classifier_pytorch.api.main:app --host 127.0.0.1 --port 8000

# Start the frontend
npm --prefix frontend run dev
```

## FFNN

```
┌─────────────────────┐
| 784 inputs (28x28)  |
└─────────────────────┘
       ↓                (784 → 128)
┌─────────────────────┐
│ Linear (128 neurons)│
│ ReLU activation     │
└─────────────────────┘
       ↓                (128 → 10)
┌─────────────────────┐
│ Linear (10 neurons) │
└─────────────────────┘
```

The 28×28 image is flattened into 784 input values. The hidden layer is a **128-neuron linear layer** followed by a **ReLU activation**. The output layer is a **10-neuron linear layer** that produces one score (logit) for each digit from 0 to 9.

## Chronological Understanding

A FFNN (feed forward neural network) is where information flows in a single direction, from the **input layer** through any **hidden layers** to the **output layer**. Our network places a nonlinear activation function between two linear layers.

```
Linear → ReLU → Linear
```
