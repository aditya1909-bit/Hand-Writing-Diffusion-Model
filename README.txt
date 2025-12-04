# Conditional Handwriting Diffusion Model

This project implements a **Conditional Diffusion Model** capable of generating handwriting that mimics a specific style based on text input. It utilizes Hugging Face's `diffusers` library for the UNet backbone and `transformers` for text encoding.

## Architecture

The system consists of three main components defined in `model.py`:

1.  **Text Encoder:** A frozen `bert-base-uncased` model that converts input text into embeddings.
2.  **Style Encoder:** A custom CNN (Conv2d layers with Batch Normalization and LeakyReLU) that encodes a reference handwriting image into a 512-dimensional style vector.
3.  **Diffusion Model:** A `UNet2DConditionModel` with cross-attention that denoises latents into handwritten text images, guided by the text and style embeddings. It uses a `DDPMScheduler` for the noise schedule.

## Prerequisites

* Python 3.8+
* PyTorch (CUDA, MPS, or CPU supported).

## Installation

1.  Clone the repository.
2.  Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt

## Dataset Setup
This project is designed to work with the IAM Handwriting Database.
1.  Download the IAM dataset (specifically words.tgz and words.txt)
2. Create a directory names iam_Data in the project repository
3. Extract the data so the structure matches the logic in data_loader.py

Project/
├── iam_data/
│   ├── words.txt           # The transcription labels file
│   └── words/              # Folder containing subfolders (a01, a02, etc.)
│       ├── a01/
│       │   ├── a01-000u/
│       │   │   └── ...png
│       └── ...


##Usage

1.  Data Inspection
        Before training verify the dataset is loading correctly and that the text-image pairs match using the inspection script.
        Output will save a composite image data_inspection.png showing 5 random samples with their decoded BERT text labels.
        python inspect_data.py

2.  Training
        The training script is expected to be run in a jupyer notebook enviornemnt.
        Ensure that HandwritingDiffusionSystem is initialized.
        The get_dataloader function in data_loader.py handels batching and can operate in "Mock Mode" for testing without a full dataset.
        Checkpoints should be saved to a ./saved_models directory.
        python train.py

3.  Generation
        The generate.py script is an interactive tool that allows you to generate handwritting using your trained checkpoints.
        It supports CUDA and Apple Silicon acceleration
        python generate.py

##How to use

Enter text you want the model to write
Enter style
        