import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import os
import math
from preprocess import get_tf_data_generators

# Configuración
MODELS_PATH = "models/pretrained/"
RESULTS_PATH = "results/pretrained/resnet50/"
EPOCHS_INITIAL = 30
EPOCHS_FINE_TUNE = 50
BATCH_SIZE = 128

# Asegurar carpetas
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Cargar datos
train_gen, val_gen, _ = get_tf_data_generators("resnet50")
num_classes = len(train_gen.class_indices)

# Cargar modelo base
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Fase 1: congelado

# Añadir capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Scheduler de learning rate
def lr_scheduler(epoch, lr):
    if epoch < EPOCHS_INITIAL:
        return 0.0001
    else:
        return 1e-5 * math.exp(-0.1 * (epoch - EPOCHS_INITIAL))

# Compilar modelo
model.compile(
    optimizer=AdamW(learning_rate=0.0001, weight_decay=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max")
checkpoint = ModelCheckpoint(
    os.path.join(MODELS_PATH, "resnet50_sign_language_best.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
lr_schedule = LearningRateScheduler(lr_scheduler)

# Fase 1: Entrenamiento con capas congeladas
print("Fase 1: Entrenando con capas congeladas...")
history_phase1 = model.fit(
    train_gen,
    epochs=EPOCHS_INITIAL,
    validation_data=val_gen,
    callbacks=[early_stopping, checkpoint, reduce_lr, lr_schedule]
)

# Fase 2: Fine-tuning
print("Fase 2: Fine-tuning...")
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=val_gen,
    callbacks=[early_stopping, checkpoint, reduce_lr, lr_schedule]
)

# Guardar modelo final
model.save(os.path.join(MODELS_PATH, "resnet50_sign_language_final.h5"))
