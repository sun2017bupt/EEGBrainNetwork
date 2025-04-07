# Anonymous EEG Emotion Recognition (Code for Paper Submission)

This repository contains the source code, experiment logs, and results for our anonymous paper on EEG-based emotion recognition. The proposed model is designed for subject-dependent emotion classification using spatial-temporal modeling of EEG phase-based brain networks.

## 📁 Directory Overview

- `home/sjf/eegall/Abla/*.out`  
  Contains training logs and prediction results for ablation studies and baseline comparisons.

- `home/sjf/eegall/withlimits/*.txt`  
  Includes text-based output logs of final prediction results under limited settings.

- `home/sjf/eegall/data/`  
  Stores essential data files including preprocessed EEG signals, selected frequency bands, and intermediate variables used during model construction.

- `home/sjf/eegall/dataex/`  
  Contains data used for plotting along with all visualization scripts and variable exports.

- `home/sjf/eegall/intermodel/`  
  Stores trained model checkpoint files and related parameter configurations.

- `home/sjf/eegall/lossre/`  
  Records the training loss values throughout the learning process.

- `home/sjf/eegall/predictionreswithdifferentparams/`  
  Contains model prediction results generated using different parameter settings for comparative analysis.

- `home/sjf/eegall/searchhyper/`  
  Contains `.sh` scripts used for automated hyperparameter search and batch experiments.

- **Core Scripts in `home/sjf/eegall/`**  
  This directory also contains the core implementation, including data preprocessing, model definition, training scripts, and evaluation tools:
  - `main.py`: The main entry point for training and evaluation.
  - `model.py`, `tgmodel.py`, `valmodel.py`, `fremodel.py`: Core model architectures.
  - `dataconstrut.py`, `EEGData.py`, `faceddata.py`: EEG data loading and preprocessing modules.
  - `customloss.py`: Custom loss functions used during training.
  - `analysismodel.ipynb`, `explainmodel.ipynb`: Jupyter notebooks for model analysis and interpretability.
  - `requirements.txt`: Python package dependencies.

## 🧪 Experimental Settings

We focus on **subject-dependent** EEG emotion recognition. Due to varying sampling durations across datasets, we adopt dataset-specific batch sizes:

- **DEAP**: batch size = 38
- **FACED**: batch size = 11

Other key settings:

- Learning rate: `5e-5`
- Optimizer: Adam
- Model parameters are initialized using **Xavier uniform initialization**
- No specific random seed is set

All experiments were conducted on an **NVIDIA A40 GPU server** with **Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz**, using **PyTorch 1.11.0**.

## 📚 Datasets

We use two publicly available EEG datasets widely used in affective computing:

- **DEAP Dataset** [[Koelstra et al., 2011]](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/):
    
    A multimodal dataset containing EEG and peripheral physiological signals collected from 32 subjects while watching 40 emotion-eliciting music videos. Emotional ratings include arousal, valence, dominance, liking, and familiarity.
    
- **FACED Dataset** [[Chen et al., 2023]](https://www.synapse.org/Synapse:syn50614194/wiki/620378):
    
    A large-scale EEG emotion dataset collected from 123 participants watching 28 emotional video clips. It includes labels across 9 emotion categories (e.g., joy, fear, sadness), as well as arousal, valence, familiarity, and liking scores.
    

> Note: Please refer to the official dataset pages to request access or download:
> 
> - DEAP: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
> - FACED: https://www.synapse.org/Synapse:syn50614194/wiki/620378

## 🔒 Note

To preserve anonymity during the review process, author information and affiliations have been removed from this repository. Upon paper acceptance, we will provide full documentation and data access instructions.
