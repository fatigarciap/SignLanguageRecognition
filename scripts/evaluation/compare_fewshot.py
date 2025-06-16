import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# Configurar directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results", "few_shot")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Datos reales de precisión por episodio
precisions = {
    "Prototypical Network": [0.3200, 0.2800, 0.2900, 0.3600, 0.4800, 0.3600, 0.5600, 0.6600, 0.4800, 0.4332],
    "Transformer Contrastive": [0.0000] * 5,
    "Matching Network": [0.0000] * 10 + [0.1667] * 5,
    "Relation Network": [0.8888] * 10
}

# Precisiones promedio
models = ["Prototypical Network", "Transformer Contrastive", "Matching Network", "Relation Network"]
train_precisions = [0.4332, 0.0213, 0.8261, 0.8888]
val_precisions = [0.6560, 0.0267, 0.8583, 0.8891]

# Asignar colores distintos para cada modelo en el Joy Plot
colors = ["blue", "red", "green", "orange"]

# Joy Plot
plt.figure(figsize=(12, 6))
sns.set_style("white")
for i, (model, prec) in enumerate(precisions.items()):
    sns.kdeplot(data=prec, label=model, fill=True, alpha=0.5, bw_adjust=0.5, warn_singular=False, color=colors[i])
plt.xlabel("Precisión")
plt.ylabel("Densidad")
plt.title("Distribución de Precisión por Episodio (Joy Plot)")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "joy_plot_accuracy.png"))
plt.close()

# Radar Chart
plt.figure(figsize=(8, 8))
angles = np.linspace(0, 2 * np.pi, 2, endpoint=False)  # Dos puntos: Entrenamiento, Validación
angles = np.concatenate((angles, [angles[0]]))  # Cerrar el polígono
values = [np.array([train, val, train]) for train, val in zip(train_precisions, val_precisions)]  # Un array por modelo
ax = plt.subplot(111, polar=True)
for model, val, color in zip(models, values, colors):
    ax.plot(angles, val, label=model, linewidth=2, color=color)
    ax.fill(angles, val, alpha=0.25, color=color)
ax.set_xticks(np.linspace(0, 2 * np.pi, 2, endpoint=False))
ax.set_xticklabels(["Entrenamiento", "Validación"])
plt.title("Comparación de Precisión (Radar Chart)")
plt.legend(bbox_to_anchor=(1.1, 1.1))
plt.savefig(os.path.join(RESULTS_DIR, "radar_chart_accuracy.png"))
plt.close()

# Tabla comparativa
data = {
    "Modelo": models,
    "Precisión Entrenamiento": train_precisions,
    "Precisión Validación": val_precisions,
    "Fortalezas": [
        "Buena generalización (validación > entrenamiento)",
        "Bajo rendimiento, pero simple de implementar",
        "Alto rendimiento y estabilidad",
        "Mejor rendimiento general y consistencia"
    ],
    "Debilidades": [
        "Precisión de entrenamiento baja",
        "Rendimiento extremadamente bajo",
        "Inconsistencia en episodios (0.0000)",
        "Requiere más datos para confirmar robustez"
    ],
    "Recomendación": [
        "Mejorar hiperparámetros",
        "Reevaluar arquitectura o datos",
        "Ideal para producción con ajustes",
        "Mejor opción para el proyecto"
    ]
}
df = pd.DataFrame(data)
df["Mejor Modelo"] = df["Precisión Validación"].idxmax() == df.index
df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
print("Tabla y gráficos guardados en:", RESULTS_DIR)