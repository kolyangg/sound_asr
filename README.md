# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

## Installation

Download repo:
```bash
   git clone https://github.com/kolyangg/sound_asr.git
   cd sound_asr
```

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   o. `python 3.12` install (if needed)

   ```bash
   apt update && apt upgrade -y
   apt install software-properties-common -y
   add-apt-repository ppa:deadsnakes/ppa
   apt update
   apt install python3.12
   python3.12 --version
   ```
   
   o. install linux dependencies (needed for kenlm)

   ```bash
   sudo apt-get update
   sudo apt-get install cmake
   sudo apt-get install build-essential
   sudo apt-get install libboost-all-dev
   ```

   
   a. `conda` version:

   ```bash
   # create env
   export PYTHON_VERSION="3.12.7"
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source activate base
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

3. Run script to download / setup LM assets and noise data:
   ```bash
   ./LM_setup.sh
   ```
   
4. Download checkpoint for the best model and sentencepiece model for BPE:
   ```bash
   python3 src/utils/checkpoint_dl.py
   ```
   
5. Login to wandb and provide API key when prompted:
   ```bash
   wandb login
   ```


## How To Use

To train the best model, run the following command:

```bash
python3 train.py -cn=big_bpe
```


To run inference on the best model, run the following command:

```bash
python3 inference.py -cn=inference_big_bpe
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
