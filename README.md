# live speech-2-text recognition

## Project Description
**live speech-2-text recognition** is a real-time, live caption speech-to-text recognition system that uses Wav2Vec model fine-tuned with the LJSpeech dataset. This project is designed to provide highly accurate live transcriptions, making it ideal for accessibility features, live events, and many other applications where real-time captioning is essential.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Transcription and Evaluation](#transcription-and-evaluation)
- [Real-time Speech Recognition](#real-time-speech-recognition)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation
To set up the project, follow these steps:

1. **Clone the repository and install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2. **Download the Model and Processor**
    - Download the fine-tuned model [here](https://vault.sfu.ca/index.php/s/y3DWNnmetsWdtnZ) and the processor files [here](https://vault.sfu.ca/index.php/s/v9vQ07HWrCOR2r3).
    - Place the downloaded folders in the `src/data/` directory. Rename the folders to `model` and `processor` respectively.

## Usage

### Model Training
If you prefer to train the model yourself:
```bash
cd src
python3 train.py
```
This script will train the Wav2Vec model using the LJSpeech dataset and save the fine-tuned model in the specified directory.

### Transcription and Evaluation
To process transcriptions and evaluate them:
- **Process Transcriptions**
    ```bash
    python3 process.py
    ```
    Use this script to process the transcriptions using the default and fine-tuned models.

- **Evaluate Transcriptions**
    ```bash
    python3 check.py
    ```
    Run this script to calculate the BLEU, WER, and CER scores for the transcriptions placed in the `src/` directory.

### Real-time Speech Recognition
For live speech recognition:
```bash
python3 speech.py
```

Ensure your microphone is set up and calibrated when prompted. Speak after the "Speak now!" prompt for live captioning.

## Acknowledgments
- Facebook AI Research for the Wav2Vec 2.0 model.
- The LJSpeech dataset contributors.

