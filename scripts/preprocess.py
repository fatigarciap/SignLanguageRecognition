import os
import shutil
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Configuración general
DATASET_PATH ="C:\\Users\\lenovo\\Desktop\\dataset_R_G_blur"
OUTPUT_PATH = "data/"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
BATCH_SIZE = 64

# Tamaños de entrada por modelo
MODEL_SIZES = {
    "mobilenet": (224, 224),
    "efficientnet": (224, 224),
    "vit": (384, 384)
}

# Dividir dataset en train, test, validation
def create_train_test_val_split():
    if not os.path.exists(OUTPUT_PATH):
        raise Exception("La carpeta data/ no existe. Crea la estructura manualmente.")
    if not os.path.exists(os.path.join(OUTPUT_PATH, "train")):
        raise Exception("La carpeta data/train/ no existe.")
    if not os.path.exists(os.path.join(OUTPUT_PATH, "validation")):
        raise Exception("La carpeta data/validation/ no existe.")
    if not os.path.exists(os.path.join(OUTPUT_PATH, "test")):
        raise Exception("La carpeta data/test/ no existe.")

    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    for class_name in classes:
        os.makedirs(os.path.join(OUTPUT_PATH, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, "validation", class_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, "test", class_name), exist_ok=True)
        
        images = [f for f in os.listdir(os.path.join(DATASET_PATH, class_name)) if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        
        train_idx = int(len(images) * TRAIN_SPLIT)
        val_idx = int(len(images) * (TRAIN_SPLIT + VAL_SPLIT))
        
        train_images = images[:train_idx]
        val_images = images[train_idx:val_idx]
        test_images = images[val_idx:]
        
        for img in train_images:
            shutil.copy(os.path.join(DATASET_PATH, class_name, img), os.path.join(OUTPUT_PATH, "train", class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(DATASET_PATH, class_name, img), os.path.join(OUTPUT_PATH, "validation", class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(DATASET_PATH, class_name, img), os.path.join(OUTPUT_PATH, "test", class_name, img))
        
        print(f"Clase {class_name}: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")

# Generador para TensorFlow con ajuste por modelo
def get_tf_data_generators(model_type):
    if model_type not in MODEL_SIZES:
        raise ValueError(f"model_type debe ser 'mobilenet', 'efficientnet' o 'vit', no '{model_type}'")
    
    img_size = MODEL_SIZES[model_type]
    
    # Aumentación adaptada por modelo
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10 if model_type != "vit" else 5,  # Menos rotación para ViT
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
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

if __name__ == "__main__":
    create_train_test_val_split()
    for model in ["mobilenet", "efficientnet", "vit"]:
        train_gen, val_gen, test_gen = get_tf_data_generators(model_type=model)
        print(f"Clases para {model}: {train_gen.class_indices}")