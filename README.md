# VocalNetSDR 🎶🔍

![Image Description](https://i.imgur.com/App5EK9.png "Image")

## Description

VocalNetSDR is a Python-based tool designed for audio processing enthusiasts and researchers 🎧. This project allows users to compare the effectiveness of different neural network models on audio tracks by calculating the Signal-to-Distortion Ratio (SDR). It specifically focuses on evaluating how well these models can separate vocals from mixed audio tracks.

The project takes advantage of various Python libraries such as NumPy, SciPy, SoundFile, and tqdm to process audio files, apply filters, and compute SDR metrics. It's highly suitable for those looking to benchmark the performance of neural network models designed for tasks such as vocal extraction, noise reduction, or any form of audio signal processing.

### Key Features

- **Model Comparison:** Dynamically compare multiple neural network models based on their SDR performance on a given dataset of audio tracks 📊.
- **Customizable Processing:** Supports custom audio processing models by specifying their names at runtime, making it highly adaptable to different models and use cases 🔧.
- **Parallel Processing:** Utilizes Python's `ThreadPoolExecutor` for concurrent processing, significantly speeding up the computation over large datasets 🚀.
- **Progress Tracking:** Integrates `tqdm` for real-time progress updates during the processing of audio tracks 📈.

## Installation

1. **Clone the repository to your local machine: 🖥️**

   ```bash
   git clone https://github.com/beatloo-lab/VocalNetSDR.git
   ```

2. **Navigate to the project directory: 📂**

   ```bash
   cd VocalNetSDR
   ```

3. **Install the dependencies: 📦**

   ```bash
   python -m pip install -r requirements.txt
   ```

## How to Use

1. **Prepare your dataset** by organizing your audio tracks into the project's `songs` directory. Each song should be in its own subdirectory, named as per your choice (e.g., `song1`). Within each song's folder, include the original mixed track and the outputs from your models. The expected file naming convention is as follows:

   - `original_other.wav`: The original background/instrumental track 🎸.
   - `original_vocals.wav`: The original vocals track 🎤.
   - `vocals_<model_name>.wav`: The separated vocals track processed by the model. Replace `<model_name>` with the name of your neural network model. This should match the names provided through the `--models` argument 🧠.

2. **Run the script** with the desired models as arguments. For example, to compare models named "bsroformer" and "instvoc", you would run ⚙️:

   ```bash
   python sdr_model_comparison.py --models bsroformer instvoc
   ```

   You can specify any number of models by adding them to the `--models` argument list, separated by spaces.

## Project Structure 🏗️

```
VocalNetSDR/
│
├── songs/                        # Directory for songs to be processed 📁
│   └── song1/                    # Each song in its own directory
│       ├── original_other.wav    # Original background/instrumental track
│       ├── original_vocals.wav   # Original vocals track
│       ├── vocals_bsroformer.wav # Processed vocals track by "bsroformer" model
│       └── vocals_instvoc.wav    # Processed vocals track by "instvoc" model
│
├── sdr_model_comparison.py       # Main script for running the comparisons 📜
└── requirements.txt              # Python dependencies 📚
```

Remember, the filenames for processed vocals tracks should follow the pattern `vocals_<model_name>.wav`, where `<model_name>` is dynamically specified via the `--models` command-line argument, allowing you to easily extend the comparison to additional models.
