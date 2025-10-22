# Lung Cancer Detection using Convolutional Neural Network (CNN)

**Author:** https://github.com/MadhumithraA1426
**Project Type:** Deep Learning / Computer Vision

---

## Description

This project implements a Convolutional Neural Network (CNN) in Python using TensorFlow and Keras to classify histopathological lung tissue images into three categories:
- **Normal lung tissue**
- **Lung Adenocarcinoma**
- **Lung Squamous Cell Carcinoma**

The model is trained and evaluated on the publicly available Lung and Colon Cancer Histopathological Images dataset.

---

## Features

- End-to-end data loading, preprocessing, and visualization
- Convolutional Neural Network for multi-class tissue classification
- Training and validation accuracy visualization
- Detailed classification report
- Easily extensible for further model tuning or deployment

---

## Installation

1. **Clone the repository:**
   git clone https://github.com/MadhumithraA1426/Lung-cancer-detection
   cd Lung-cancer-detection
2. **Install dependencies:**  
   pip install -r requirements.txt

---

## Dataset

- Download the [Lung and Colon Cancer Histopathological Images Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) from Kaggle.
- Place the unzipped `lung_colon_image_set` folder in your project directory.

*Note: The dataset is large and should not be uploaded to GitHub. Only use a local copy for training.*

---

## Usage

1. **Run the Python script:**
     python lung_cancer_detection.py

2. **Model Training:**  
Visualizes sample images, preprocesses the dataset, trains the CNN, plots training/validation accuracy, and prints the classification report.

3. **Model Saving:**  
The trained model is saved as `lung_cancer_cnn_model.h5` (or you can use Keras format: `.keras`).

---

## Results

Achieves high accuracy on normal tissue, with room for improvement on adenocarcinoma and squamous cell carcinoma.

**Example output (classification report):**
      precision recall f1-score support
      lung_n 0.91 1.00 0.95 987
      lung_aca 0.57 0.91 0.70 977
      lung_scc 0.98 0.35 0.52 1036

---

## Project Structure
     lung_cancer_detection.py
     requirements.txt
     README.md
    .gitignore
     lung_colon_image_set/ # (NOT in GitHub, kept locally)
     lung_cancer_cnn_model.h5 # (model file, kept locally)

---

## References

- GeeksforGeeks: [Lung Cancer Detection using CNN](https://www.geeksforgeeks.org/deep-learning/lung-cancer-detection-using-convolutional-neural-network-cnn/)
- Kaggle: [Lung and Colon Cancer Histopathological Images Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- TensorFlow, Keras Documentation

---

