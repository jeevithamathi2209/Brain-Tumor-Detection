"""
predict.py - Brain Tumor Detection Prediction Module
Handles image preprocessing and model inference.
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Class labels matching training dataset folders
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Human-readable display names
CLASS_DISPLAY_NAMES = {
    'glioma': 'Glioma Tumor',
    'meningioma': 'Meningioma Tumor',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary Tumor'
}

# Severity/info descriptions
CLASS_DESCRIPTIONS = {
    'glioma': 'Glioma is a tumor that starts in the glial cells of the brain or spinal cord.',
    'meningioma': 'Meningioma is a tumor that arises from the meninges surrounding the brain and spinal cord.',
    'notumor': 'No tumor detected in the MRI scan. The brain appears normal.',
    'pituitary': 'Pituitary tumor is an abnormal growth in the pituitary gland at the base of the brain.'
}

IMG_SIZE = 150  # Must match training image size


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an MRI image for model input.
    Steps: Load → Resize → Normalize → Noise Reduction → Expand dims
    """
    # Read image in color
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # Apply Gaussian blur for noise reduction
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Resize to model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0

    # Expand dimensions to match model input shape (1, H, W, C)
    img = np.expand_dims(img, axis=0)

    return img


def predict_tumor(image_path: str, model_path: str = 'models/brain_tumor_model.h5') -> dict:
    """
    Run prediction on a given MRI image.
    Returns a dictionary with prediction result, confidence, and metadata.
    """
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Please train the model first using train.py"
        )

    model = load_model(model_path)

    # Preprocess image
    processed_img = preprocess_image(image_path)

    # Run inference
    predictions = model.predict(processed_img, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100

    predicted_class = CLASS_LABELS[predicted_index]
    display_name = CLASS_DISPLAY_NAMES[predicted_class]
    description = CLASS_DESCRIPTIONS[predicted_class]

    # Build result dict
    result = {
        'class': predicted_class,
        'display_name': display_name,
        'confidence': round(confidence, 2),
        'description': description,
        'is_tumor': predicted_class != 'notumor',
        'all_probabilities': {
            CLASS_DISPLAY_NAMES[CLASS_LABELS[i]]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_LABELS))
        }
    }

    return result


if __name__ == '__main__':
    # Quick test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        result = predict_tumor(sys.argv[1])
        print(f"Prediction : {result['display_name']}")
        print(f"Confidence : {result['confidence']}%")
        print(f"Description: {result['description']}")
