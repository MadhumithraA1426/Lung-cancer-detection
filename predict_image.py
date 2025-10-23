import numpy as np
import cv2
from tensorflow import keras
import sys

# Change this if your model filename is different
MODEL_PATH = "lung_cancer_cnn_model.h5"    # or use 'lung_cancer_cnn_model.keras'

# Class names (order matches training order)
CLASS_NAMES = ['lung_n', 'lung_aca', 'lung_scc']

# Set image size as per training
IMG_SIZE = 256

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image {image_path}")
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Convert to numpy array and expand dims for batch
    img_array = np.array(img)
    img_array = img_array / 255.0   # Normalize to [0,1], optional if done in training
    img_array = np.expand_dims(img_array, axis=0)   # shape: (1, IMG_SIZE, IMG_SIZE, 3)
    return img_array

def predict(image_path):
    # Load model
    model = keras.models.load_model(MODEL_PATH)
    # Preprocess image
    img_array = preprocess_image(image_path)
    # Predict
    prediction_prob = model.predict(img_array)[0]
    predicted_idx = np.argmax(prediction_prob)
    predicted_class = CLASS_NAMES[predicted_idx]
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {prediction_prob}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py /path/to/image.jpg")
    else:
        image_path = sys.argv[1]
        predict(image_path)
