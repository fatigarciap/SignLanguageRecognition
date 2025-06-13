import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Añade el directorio raíz
from preprocess import get_tf_data_generators
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_PATH = os.path.join(BASE_DIR, "models", "pretrained")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "pretrained", "resnet50")  # Cambiado a subcarpeta mobilenet
MODEL_FILE = "resnet50_sign_language_best.h5"  # Ajusta según el modelo a evaluar
MODEL_TYPE = "resnet50"  # Define el tipo de modelo a evaluar

# Asegurarse de que las carpetas existan
os.makedirs(RESULTS_PATH, exist_ok=True)

# Función para visualizar predicciones
def visualize_predictions(model, test_gen, num_images=5):
    test_gen.reset()
    sample_indices = np.random.choice(len(test_gen.filenames), num_images, replace=False)
    sample_images = []
    sample_labels = []
    class_names = list(test_gen.class_indices.keys())
    
    for idx in sample_indices:
        img_path = test_gen.filepaths[idx]
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  # Ajusta si usas otro modelo
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        sample_images.append(img_array)
        sample_labels.append(test_gen.classes[idx])

    sample_images = np.array(sample_images)
    sample_predictions = model.predict(sample_images)
    sample_pred_labels = np.argmax(sample_predictions, axis=1)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(sample_images[i])
        true_label = class_names[sample_labels[i]]
        pred_label = class_names[sample_pred_labels[i]]
        is_correct = "Correcto" if true_label == pred_label else "Incorrecto"
        plt.title(f"Real: {true_label}\nPred: {pred_label}\n({is_correct})")
        plt.axis('off')
    plt.savefig(os.path.join(RESULTS_PATH, "sample_predictions.png"))
    plt.close()

# Cargar datos y modelo
_, _, test_gen = get_tf_data_generators(MODEL_TYPE)
model = load_model(os.path.join(MODELS_PATH, MODEL_FILE))

# Evaluar modelo
test_gen.reset()
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

# Métricas
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred)

# Guardar y mostrar resultados
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
with open(os.path.join(RESULTS_PATH, "evaluation_metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")

# Guardar matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix.png"))
plt.close()

# Visualizar predicciones
visualize_predictions(model, test_gen)