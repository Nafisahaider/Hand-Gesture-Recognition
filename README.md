# Hand-Gesture-Recognition
Overview

This project implements a hand gesture recognition model using a dataset of American Sign Language (ASL) alphabets. The model is trained on images of ASL hand gestures and is capable of classifying different gestures to enable intuitive human-computer interaction.

## Dataset

The dataset used for this project is the ASL Alphabet Dataset, which can be found on Kaggle:
ASL Alphabet Dataset

## Dataset Structure

The dataset consists of 29 classes (A-Z, space, nothing, delete).

Each class folder contains images of corresponding hand gestures.

All images are in .jpg format.

## Project Structure

Hand_Gesture_Recognition/
│── dataset/                 # Contains the ASL Alphabet dataset
│── models/                  # Saved trained models
│── notebooks/               # Jupyter notebooks for experimentation
│── src/
│   ├── data_loader.py       # Script to load and preprocess images
│   ├── model.py             # Model architecture definition
│   ├── train.py             # Training script
│   ├── test.py              # Testing and evaluation script
│── README.md                # Project documentation
│── requirements.txt         # Dependencies

## Installation

### Step 1: Clone the Repository

git clone https://github.com/your-repo/Hand_Gesture_Recognition.git
cd Hand_Gesture_Recognition

### Step 2: Install Dependencies

Ensure you have Python 3.7+ installed. Install required libraries:

pip install -r requirements.txt

### Step 3: Download the Dataset

Download the dataset from Kaggle: ASL Alphabet Dataset

Extract the dataset into the dataset/ directory.

## Usage

### 1. Data Preprocessing

Run the following script to load and preprocess the dataset:

python src/data_loader.py

### 2. Train the Model

To train the model, execute:

python src/train.py

### 3. Evaluate the Model

Run the test script to evaluate the model:

python src/test.py

## Implementation Details

### Image Preprocessing

Images are resized to 128x128 pixels.

Converted to grayscale for simplicity.

Normalized pixel values to the range [0,1].

### Model Architecture

CNN-based deep learning model implemented using TensorFlow/Keras.

Includes multiple convolutional layers, max-pooling, and fully connected layers.

Categorical cross-entropy loss with Adam optimizer.
