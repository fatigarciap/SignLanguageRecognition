import os
import shutil
import random
import tensorflow as tf

# Configuración general
DATASET_PATH = "C:\\Users\\lenovo\\Desktop\\dataset_R_G_blur"
OUTPUT_PATH = "data/"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
BATCH_SIZE = 64

# Tamaños de entrada por modelo
MODEL_SIZES = {
    "mobilenet": (224, 224),
    "efficientnet": (224, 224),
    "resnet50": (224, 224)
}

# Crear estructura si no existe
def ensure_directory_structure():
    for subset in ["train", "validation", "test"]:
        subset_path = os.path.join(OUTPUT_PATH, subset)
        os.makedirs(subset_path, exist_ok=True)

# Dividir dataset en train, test, validation
def create_train_test_val_split():
    ensure_directory_structure()

    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

    for class_name in classes:
        print(f"Procesando clase: {class_name}")
        class_dir = os.path.join(DATASET_PATH, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)

        train_idx = int(len(images) * TRAIN_SPLIT)
        val_idx = int(len(images) * (TRAIN_SPLIT + VAL_SPLIT))

        splits = {
            "train": images[:train_idx],
            "validation": images[train_idx:val_idx],
            "test": images[val_idx:]
        }

        for split, split_images in splits.items():
            dest_dir = os.path.join(OUTPUT_PATH, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img in split_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(dest_dir, img))

        print(f" - {len(splits['train'])} train | {len(splits['validation'])} val | {len(splits['test'])} test")

# Generador para TensorFlow con ajuste por modelo
def get_tf_data_generators(model_type):
    if model_type not in MODEL_SIZES:
        raise ValueError(f"model_type debe ser uno de: {list(MODEL_SIZES.keys())}, no '{model_type}'")

    img_size = MODEL_SIZES[model_type]

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
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

# Ejecutar si es principal
if __name__ == "__main__":
    create_train_test_val_split()
    for model in ["mobilenet", "efficientnet", "resnet50"]:
        train_gen, val_gen, test_gen = get_tf_data_generators(model_type=model)
        print(f"Modelo: {model} → {len(train_gen.class_indices)} clases: {train_gen.class_indices}")
