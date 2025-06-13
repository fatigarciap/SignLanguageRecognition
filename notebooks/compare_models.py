import tensorflow as tf
from tensorflow import keras
import sys
import os

# Añadir el directorio 'scripts' al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_path = os.path.join(project_root, 'scripts')
sys.path.append(scripts_path)

# Verificar si el módulo preprocess está disponible
try:
    from preprocess import get_tf_data_generators
except ModuleNotFoundError:
    print("Error: No se encontró el módulo 'preprocess'. Asegúrate de que 'preprocess.py' esté en la carpeta 'scripts'.")
    sys.exit(1)

# Configuración
MODELS_PATH = "models/pretrained/"

# Cargar el generador de validación
try:
    _, val_gen, _ = get_tf_data_generators("efficientnet")
except Exception as e:
    print(f"Error al cargar los generadores de datos: {e}")
    sys.exit(1)

# Cargar los modelos
try:
    model_best = keras.models.load_model(os.path.join(MODELS_PATH, "efficientnet_sign_language_best.h5"))
    model_final = keras.models.load_model(os.path.join(MODELS_PATH, "efficientnet_sign_language_final.h5"))
except Exception as e:
    print(f"Error al cargar los modelos: {e}")
    sys.exit(1)

# Evaluar el modelo "best"
best_eval = model_best.evaluate(val_gen, verbose=1)
print(f"Modelo 'best' - Loss: {best_eval[0]:.4f}, Accuracy: {best_eval[1]:.4f}")

# Evaluar el modelo "final"
final_eval = model_final.evaluate(val_gen, verbose=1)
print(f"Modelo 'final' - Loss: {final_eval[0]:.4f}, Accuracy: {final_eval[1]:.4f}")

# Comparar y recomendar
if best_eval[1] > final_eval[1]:
    print("El modelo 'best' es mejor (mayor accuracy).")
    # Guardar el modelo 'best' en formato .keras
    model_best.save(os.path.join(MODELS_PATH, "efficientnet_sign_language_best.keras"))
elif final_eval[1] > best_eval[1]:
    print("El modelo 'final' es mejor (mayor accuracy).")
    # Guardar el modelo 'final' en formato .keras
    model_final.save(os.path.join(MODELS_PATH, "efficientnet_sign_language_final.keras"))
else:
    print("Ambos modelos tienen el mismo accuracy. Considera comparar la loss o usar el modelo 'best'.")




