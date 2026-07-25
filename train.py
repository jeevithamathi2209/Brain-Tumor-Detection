"""
train.py - CNN Model Training Script for Brain Tumor Detection
Supports dataset structure:
    dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/

Run: python train.py
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "Training")
TEST_DIR = os.path.join(DATASET_DIR, "Testing")
MODEL_SAVE_PATH = "models/brain_tumor_model.h5"

IMG_SIZE    = 150
BATCH_SIZE  = 32
EPOCHS      = 35
RANDOM_SEED = 42
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)

np.random.seed(42)
tf.random.set_seed(42)

def load_images_from_folder(base_dir: str, split_name: str) -> tuple:
    images,labels = [],[]
    print(f"\n Loading [{split_name}] from: {base_dir}")
    
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(base_dir, class_name)

        if not os.path.exists(class_dir):
           print(f"Not found: {class_dir}")
           continue
    
        files = [
           f for f in os.listdir(class_dir)
           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]
        
        print(f"  {class_name:12s}: {len(files):>5d} images")

        for fname in files:
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.GaussianBlur(img, (3, 3), 0)       # noise reduction
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))   # resize
            images.append(img)
            labels.append(label_idx)

    images_arr = np.array(images, dtype='float32') / 255.0  # normalize [0,1]
    labels_arr = np.array(labels)
    print(f"    → Total: {len(images_arr)} images")
    return images_arr, labels_arr


def detect_dataset_structure() -> str:
    """
    Auto-detect dataset layout and return mode:
      'split'  → dataset/Training/ and dataset/Testing/ exist
      'flat'   → dataset/glioma/ etc. exist directly
      'none'   → nothing found
    """
    has_training = os.path.isdir(TRAIN_DIR)
    has_testing  = os.path.isdir(TEST_DIR)
    has_flat     = os.path.isdir(os.path.join(DATASET_DIR, 'glioma'))

    if has_training and has_testing:
        return 'split'
    elif has_training:           # only Training folder
        return 'split_train_only'
    elif has_flat:
        return 'flat'
    else:
        return 'none'


# ─── CNN Model Architecture ───────────────────────────────────────────────────
def build_model() -> Sequential:
    """Custom 4-block CNN for 4-class brain tumor classification."""
    model = Sequential([
        # Block 1 — 32 filters
        Conv2D(32, (3,3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2 — 64 filters
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3 — 128 filters
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        # Block 4 — 256 filters
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        # Classifier head
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("   Brain Tumor Detection — Model Training")
    print("=" * 60)

    # ── Detect structure ──────────────────────────────────────────
    mode = detect_dataset_structure()
    print(f"\n Dataset structure detected: [{mode}]")

    if mode == 'none':
        print("\n No dataset found!")
        print("   Expected one of:")
        print(f"     • {TRAIN_DIR}/glioma/, {TEST_DIR}/glioma/ ...")
        print(f"     • {DATASET_DIR}/glioma/ ...")
        return

    # ── Load data depending on structure ──────────────────────────
    if mode == 'split':
        # Use Training folder for train+val, Testing folder for test
        X_trainval, y_trainval = load_images_from_folder(TRAIN_DIR, 'Training')
        X_test,     y_test_raw = load_images_from_folder(TEST_DIR,  'Testing')

        if len(X_trainval) == 0:
            print(" Training folder is empty.")
            return

        # Split Training → 85% train, 15% validation
        X_train, X_val, y_train_raw, y_val_raw = train_test_split(
            X_trainval, y_trainval,
            test_size=0.15,
            random_state=RANDOM_SEED,
            stratify=y_trainval
        )

        # One-hot encode
        y_train = to_categorical(y_train_raw, NUM_CLASSES)
        y_val   = to_categorical(y_val_raw,   NUM_CLASSES)
        y_test  = to_categorical(y_test_raw,  NUM_CLASSES)

    elif mode == 'split_train_only':
        # Only Training/ exists — split it 70/15/15
        X_all, y_all = load_images_from_folder(TRAIN_DIR, 'Training')
        if len(X_all) == 0:
            print("Training folder is empty.")
            return

        X_train, X_temp, y_tr, y_temp = train_test_split(
            X_all, y_all, test_size=0.30,
            random_state=RANDOM_SEED, stratify=y_all
        )
        X_val, X_test, y_v, y_te = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED
        )
        y_train = to_categorical(y_tr,  NUM_CLASSES)
        y_val   = to_categorical(y_v,   NUM_CLASSES)
        y_test  = to_categorical(y_te,  NUM_CLASSES)

    else:  # 'flat'
        X_all, y_all = load_images_from_folder(DATASET_DIR, 'Full Dataset')
        if len(X_all) == 0:
            print(" Dataset is empty.")
            return

        X_train, X_temp, y_tr, y_temp = train_test_split(
            X_all, y_all, test_size=0.30,
            random_state=RANDOM_SEED, stratify=y_all
        )
        X_val, X_test, y_v, y_te = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED
        )
        y_train = to_categorical(y_tr,  NUM_CLASSES)
        y_val   = to_categorical(y_v,   NUM_CLASSES)
        y_test  = to_categorical(y_te,  NUM_CLASSES)

    # ── Print split summary ────────────────────────────────────────
    print(f"\n Dataset Split Summary:")
    print(f"   Train      : {len(X_train):>5d} images")
    print(f"   Validation : {len(X_val):>5d} images")
    print(f"   Test       : {len(X_test):>5d} images")
    print(f"   Total      : {len(X_train)+len(X_val)+len(X_test):>5d} images")

    # ── Data augmentation (training only) ─────────────────────────
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.05
    )

    # ── Build & summarize model ────────────────────────────────────
    model = build_model()
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        )
    ]

    # ── Train ─────────────────────────────────────────────────────
    print("\n Starting training...\n")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluate on Test set ───────────────────────────────────────
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy : {test_acc * 100:.2f}%")
    print(f"   Test Loss     : {test_loss:.4f}")

    # Classification report
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

    # ── Confusion Matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        cmap='Blues'
    )
    plt.title('Confusion Matrix — Brain Tumor Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    print("\n   Confusion matrix → models/confusion_matrix.png")

    # ── Training Curves ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'],     label='Train Acc',  color='#1a73e8')
    ax1.plot(history.history['val_accuracy'], label='Val Acc',    color='#f97316', linestyle='--')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train Loss', color='#1a73e8')
    ax2.plot(history.history['val_loss'], label='Val Loss',   color='#f97316', linestyle='--')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Brain Tumor CNN — Training History', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150)
    print("   Training curves  → models/training_curves.png")
    print(f"\n Model saved → {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    train()


