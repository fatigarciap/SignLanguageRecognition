import os
import shutil
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch
from torchvision import transforms
from pathlib import Path

# Configuración
DATASET_PATH = DATASET_PATH = "C:\\Users\\lenovo\\Desktop\\dataset_R_G_blur"
OUTPUT_PATH = "data/"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
IMG_SIZE_MOBILENET = (224, 224)
IMG_SIZE_VIT = (384, 384)
BATCH_SIZE = 32

# Crear carpetas train, test, validation
def create_train_test_val_split():
    os.makedirs(os.path.join(OUTPUT_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "test"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "validation"), exist_ok=True)

    # Obtener lista de clases (carpetas en dataset_R_G_blur)
    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    for class_name in classes:
        # Crear subcarpetas por clase
        os.makedirs(os.path.join(OUTPUT_PATH, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, "test", class_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, "validation", class_name), exist_ok=True)
        
        # Obtener imágenes
        images = [f for f in os.listdir(os.path.join(DATASET_PATH, class_name)) if f.endswith(('.jpg', '.png'))]
        random.shuffle(images)
        
        # Dividir en train, val, test
        train_idx = int(len(images) * TRAIN_SPLIT)
        val_idx = int(len(images) * (TRAIN_SPLIT + VAL_SPLIT))
        
        train_images = images[:train_idx]
        val_images = images[train_idx:val_idx]
        test_images = images[val_idx:]
        
        # Copiar imágenes a las carpetas correspondientes
        for img in train_images:
            shutil.copy(os.path.join(DATASET_PATH, class_name, img), os.path.join(OUTPUT_PATH, "train", class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(DATASET_PATH, class_name, img), os.path.join(OUTPUT_PATH, "validation", class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(DATASET_PATH, class_name, img), os.path.join(OUTPUT_PATH, "test", class_name, img))

# Generador para TensorFlow (MobileNetV2, EfficientNet)
def get_tf_data_generators(model_type="mobilenet"):
    img_size = IMG_SIZE_MOBILENET if model_type in ["mobilenet", "efficientnet"] else IMG_SIZE_VIT
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(OUTPUT_PATH, "train"),
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(OUTPUT_PATH, "validation"),
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    
    test_generator = val_datagen.flow_from_directory(
        os.path.join(OUTPUT_PATH, "test"),
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

# Transformaciones para PyTorch (Vision Transformer, few-shot)
def get_torch_transforms(model_type="vit"):
    img_size = IMG_SIZE_VIT if model_type == "vit" else IMG_SIZE_MOBILENET
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    create_train_test_val_split()
    # Ejemplo: Generar datos para MobileNet
    train_gen, val_gen, test_gen = get_tf_data_generators(model_type="mobilenet")
    print(f"Clases encontradas: {train_gen.class_indices}")