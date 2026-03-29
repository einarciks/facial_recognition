# Face Expression Recognition using CNN

## Overview

This project implements a **Face Expression Recognition system** using a Convolutional Neural Network (CNN).
The model classifies facial images into different emotion categories such as:

* Angry
* Happy
* Neutral
* Sad

The system is capable of:

* Training a CNN model on image datasets
* Predicting emotions from images
* Performing real-time emotion detection using a webcam

---


## Installation

1. Clone the repository:

```
git clone https://github.com/einarciks/facial_recognition
```
---

## Dataset

The dataset should be organized into folders by emotion classes.

Example:

```
dataset/train/happy/
dataset/train/sad/
```
---

## Training the Model

Run the training script:

```
train.py
```

After training, the model will be saved as:

```
emotion_model.pth
```

---

## Real-Time Emotion Detection

Run:

```
recognition.py
```

This will:

* Open your webcam
* Detect faces
* Display predicted emotion in real time

Press **Q** to quit.

---

## Model Architecture

The CNN consists of:

* 3 Convolutional layers
* ReLU activation
* MaxPooling layers
* Fully connected layers
* Dropout for regularization
---
