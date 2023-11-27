# Early_Weight_Avg
Pre-train Large Language Models (LLMs) faster with Early Weight Averaging. This repository is currently a work in progress.

## Abstract
Training Large Language Models (LLMs) incurs significant cost, making strategies that accelerate model convergence highly valuable. In our research, we focus on the impact of checkpoint averaging along the trajectory of a training run to enhance both convergence and generalization early in the training process. We observe that models trained with high learning rates benefit more from checkpoint averaging. This effect is further intensified when checkpoints are sampled with substantial spacing in training steps. Our training method surpasses conventional training and popular checkpoint averaging baselines such as exponential moving average (EMA) and stochastic moving average (SWA). We demonstrate the efficacy of our approach by pre-training nanoGPT-2 models of various sizes—small (125M), medium (335M), and large (770M)—on the OpenWebText dataset, consisting of 9 billion tokens. We also present results for publicly available Pythia LLMs, ranging from 1 billion to 12 billion parameters, trained on the PILE-deduped dataset containing 207 billion tokens.

## Training Script for Small nanoGPT-2
To train a small nanoGPT-2 model, use the following command:
```bash
torchrun --standalone --nproc_per_node=3 train_ema.py config/train_gpt2_small_adam.py

