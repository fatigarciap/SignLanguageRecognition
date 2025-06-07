import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
from preprocess import get_tf_data_generators

# Configuración
MODELS_PATH = "models/pretrained/"
RESULTS_PATH = "results/pretrained/"
EPOCHS = 10

# Asegurarse de que las carpetas existan
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Cargar datos
train_gen, val_gen, test_gen = get_tf_data_generators("mobilenet")
num_classes = len(train_gen.class_indices)

# Cargar modelo preentrenado
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelar capas base

# Añadir capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar y entrenar
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

# Evaluar
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"MobileNetV2 Test Accuracy: {test_accuracy:.4f}")

# Guardar modelo
model.save(os.path.join(MODELS_PATH, "mobilenet_sign_language.h5"))

# Guardar resultados
with open(os.path.join(RESULTS_PATH, "comparison.txt"), "w") as f:
    f.write(f"MobileNetV2 Accuracy: {test_accuracy:.4f}\n")