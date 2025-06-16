import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

#Definir el modelo Prototypical Network
class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 64)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# Función para crear un episodio
def create_episode(dataset, n_way=3, n_shot=10, n_query=5, device="cpu"):
    class_names = dataset.classes
    all_indices = list(range(len(dataset)))
    selected_classes = np.random.choice(len(class_names), n_way, replace=False)
    
    support_set = []
    query_set = []
    support_labels = []
    query_labels = []

    for i, class_idx in enumerate(selected_classes):
        class_indices = [idx for idx in all_indices if dataset.targets[idx] == class_idx]
        np.random.shuffle(class_indices)
        support_indices = class_indices[:n_shot]
        query_indices = class_indices[n_shot:n_shot + n_query]

        for idx in support_indices:
            img, _ = dataset[idx]
            support_set.append(img)
        for idx in query_indices:
            img, _ = dataset[idx]
            query_set.append(img)
        support_labels.extend([i] * n_shot)
        query_labels.extend([i] * n_query)

    support = torch.stack(support_set).to(device)
    query = torch.stack(query_set).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
    query_labels = torch.tensor(query_labels, dtype=torch.long).to(device)
    return support, query, support_labels, query_labels

# Función para desnormalizar imágenes
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Desnormalizar: (tensor * std) + mean
    return torch.clamp(tensor, 0, 1)  # Asegurar rango [0, 1]

# Función para evaluar el modelo con matriz de confusión y predicciones de muestra
def evaluate_few_shot(model, dataset, results_path, n_way=3, n_shot=10, n_query=5, n_episodes=50, device="cpu", num_images=5):
    model.eval()
    all_predictions = []
    all_labels = []
    class_names = dataset.classes

    with torch.no_grad():
        for _ in range(n_episodes):
            support, query, support_labels, query_labels = create_episode(dataset, n_way, n_shot, n_query, device)

            scores = model(support, query)
            scores = scores.mean(dim=2)
            _, predictions = torch.max(scores, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Predicciones de muestra
    sample_indices = np.random.choice(len(dataset), num_images, replace=False)
    sample_images = []
    sample_labels = []
    for idx in sample_indices:
        img, label = dataset[idx]
        img = img.unsqueeze(0).to(device)
        sample_images.append(img)
        sample_labels.append(label)

    sample_images = torch.cat(sample_images, dim=0)
    support, _, support_labels, _ = create_episode(dataset, n_way, n_shot, 0, device)
    scores = model(support, sample_images)
    scores = scores.mean(dim=2)
    _, sample_predictions = torch.max(scores, dim=1)

    # Visualizar predicciones
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        img = sample_images[i].cpu().permute(1, 2, 0).numpy()  # Convertir a formato HWC
        img = denormalize(torch.from_numpy(img).float())  # Desnormalizar
        plt.imshow(img)
        true_label = class_names[sample_labels[i]]
        pred_label = class_names[sample_predictions[i].item()]
        is_correct = "Correcto" if true_label == pred_label else "Incorrecto"
        plt.title(f"Real: {true_label}\nPred: {pred_label}\n({is_correct})")
        plt.axis('off')
    plt.savefig(os.path.join(results_path, "prototypical_sample_predictions.png"))
    plt.close()

    # Guardar matriz de confusión como heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Prototypical Network")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(results_path, "confusion_matrix.png"))
    plt.close()

    # Guardar métricas y predicciones
    with open(os.path.join(results_path, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Model: Prototypical Network\n")
        f.write(f"n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, n_episodes: {n_episodes}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

    with open(os.path.join(results_path, "sample_predictions.txt"), "w") as f:
        f.write(f"Predicciones de muestra para Prototypical Network:\n")
        for i in range(num_images):
            true_label = class_names[sample_labels[i]]
            pred_label = class_names[sample_predictions[i].item()]
            is_correct = "Correcto" if true_label == pred_label else "Incorrecto"
            f.write(f"Imagen {i}: Real: {true_label}, Pred: {pred_label}, {is_correct}\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, f1, conf_matrix

# Configuración principal
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transformaciones para las imágenes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cargar datos de validación
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    val_dataset = ImageFolder(os.path.join(BASE_DIR, "data", "validation"), transform=transform)

    # Configurar modelo Relation Network
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = RelationNetwork(backbone).to(device)  # Usar el backbone proporcionado
    model_path = os.path.join(BASE_DIR, "models", "few_shot", "prototypical_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Definir ruta de resultados
    RESULTS_PATH = os.path.join(BASE_DIR, "results", "few_shot", "prototypical")
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Evaluar el modelo
    n_way = 3
    n_shot = 10
    n_query = 5
    n_episodes = 50
    num_images = 5

    accuracy, f1, conf_matrix = evaluate_few_shot(model, val_dataset, RESULTS_PATH, n_way, n_shot, n_query, n_episodes, device, num_images)

if __name__ == "__main__":
    main()