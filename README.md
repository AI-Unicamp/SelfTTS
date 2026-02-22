<div align="center">

# SelfTTS

**Cross-speaker style transfer through explicit embedding disentanglement and self-refinement using self-augmentation**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![SLURM](https://img.shields.io/badge/SLURM-HPC%20Ready-green?style=flat-square)](https://slurm.schedmd.com)
[![License](https://img.shields.io/badge/License-Research-orange?style=flat-square)]()

*Official implementation of the SelfTTS paper*

</div>

---

## ✨ Overview

SelfTTS is a text-to-speech system designed for **cross-speaker style transfer** — enabling the voice characteristics and emotional style of one speaker to be applied to another. It achieves this through two key innovations:

- **Explicit Embedding Disentanglement** — cleanly separating speaker identity from speaking style in the latent space
- **Self-Refinement via Self-Augmentation** — iteratively improving output quality using its own generated data as training signal

---

## 📦 Dataset

We use the [ESD (Emotional Speech Dataset)](https://github.com/HLTSingapore/Emotional-Speech-Data) for training. A download and formatting script is provided for convenience.

```bash
sh download_esd.sh
```

> ⚠️ **Note:** If you use the original ESD data format instead of the one produced by our script, you will need to adapt the filelists accordingly.

---

## 🛠️ Environment Setup

We provide a setup script that assumes a **Conda** installation. It will automatically create a new environment named `selftts` and install all dependencies from `requirements.txt`.

```bash
sh make_selftts_env.sh
```

Feel free to adapt the environment configuration to your own needs.

---

## 🚀 Training

### Step 1 — Download the Base VCTK Model

```bash
mkdir vctk_base_16k
wget https://github.com/AI-Unicamp/SelfTTS/releases/download/v1.0.0/G_800000.pth -O vctk_base_16k/G_800000.pth
wget https://github.com/AI-Unicamp/SelfTTS/releases/download/v1.0.0/D_800000.pth -O vctk_base_16k/D_800000.pth
```

---

### Step 2 — Train SelfTTS

We provide a SLURM script for HPC environments:

```bash
sbatch run_selftts.sh
```

Or run manually:

```bash
source ~/miniconda3/bin/activate
conda activate selftts

# Link your ESD dataset
rm -rf DUMMY3
ln -s {your_esd_base_path} DUMMY3

# Launch training
python train_ms_emotion.py -c configs/selftts_training.json -m selftts_training
```

---

### Step 3 — Train Self-Refinement with Self-Augmentation

Ensure to give the right path at the corresponding config file. If you want to train only the self-augmentation step we provide a checkpoint of SelfTTS:


```bash
mkdir logs/
mkdir logs/selftts
wget https://github.com/AI-Unicamp/SelfTTS/releases/download/v1.0.0/G_200000.pth -O logs/selftts/G_200000.pth
wget https://github.com/AI-Unicamp/SelfTTS/releases/download/v1.0.0/D_200000.pth -O logs/selftts/D_200000.pth
```


We provide a dedicated SLURM script:

```bash
sbatch run_selftts_selfaugmentation.sh
```

Or run manually:

```bash
source ~/miniconda3/bin/activate
conda activate selftts

# Link your ESD dataset
rm -rf DUMMY3
ln -s {your_esd_base_path} DUMMY3

# Launch self-augmentation training
python train_ms_emotion_selfaug.py -c configs/selftts_selfaugmentation.json -m selftts_selfaugmentation
```

---

## 🔗 Acknowledgements

This work builds upon and is inspired by the following open-source projects:

| Project | Description |
|---|---|
| [VITS](https://github.com/jaywalnut310/vits) | End-to-end TTS backbone |
| [syn-rep-learn](https://github.com/google-research/syn-rep-learn) | Synthetic representation learning |
| [Coqui TTS](https://github.com/coqui-ai/TTS) | TTS toolkit reference |

---

## 📄 Citation

> Citation coming soon — paper under review.

```bibtex
@article{selftts,
  title   = {SelfTTS: Cross-Speaker Style Transfer through Explicit Embedding
             Disentanglement and Self-Refinement using Self-Augmentation},
  author  = {},
  year    = {2024},
  note    = {Coming soon}
}
```
