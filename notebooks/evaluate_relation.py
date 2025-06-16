
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

# Definir el modelo Relation Network
class RelationNetwork(nn.Module):
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, support, query):
        batch_size = query.size(0)
        n_support = support.size(0)
        support_features = self.backbone(support).view(n_support, -1)
        query_features = self.backbone(query).view(batch_size, -1)
        support_expanded = support_features.unsqueeze(0).repeat(batch_size, 1, 1)
        query_expanded = query_features.unsqueeze(1).repeat(1, n_support, 1)
        relations = torch.cat((support_expanded, query_expanded), dim=-1)
        scores = self.fc(relations).view(batch_size, n_support)
        return scores

# Función para desnormalizar imágenes
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

# Función para crear un episodio
def create_episode(dataset, n_way=3, n_shot=1, n_query=1, device="cpu"):
    class_names = dataset.classes
    all_indices = list(range(len(dataset)))
    class_counts = np.bincount(dataset.targets)
    valid_classes = [i for i, count in enumerate(class_counts) if count >= n_shot + n_query]
    if len(valid_classes) < n_way:
        print(f"Error: Solo {len(valid_classes)} clases tienen suficientes imágenes (>= {n_shot + n_query})")
        return None, None, None, None
    
    selected_classes = np.random.choice(valid_classes, n_way, replace=False)
    
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

    if not support_set or not query_set:
        print("Error: No se pudieron crear conjuntos de soporte o consulta")
        return None, None, None, None
    support = torch.stack(support_set).to(device)
    query = torch.stack(query_set).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
    query_labels = torch.tensor(query_labels, dtype=torch.long).to(device)
    return support, query, support_labels, query_labels

# Función para evaluar el modelo
def evaluate_few_shot(model, dataset, results_path, n_way=3, n_shot=1, n_query=1, n_episodes=20, device="cpu", num_images=3):
    model.eval()
    all_predictions = []
    all_labels = []
    class_names = dataset.classes
    os.makedirs(results_path, exist_ok=True)

    # Transformación para imágenes crudas
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    raw_dataset = ImageFolder(os.path.join(os.path.dirname(results_path), "..", "..", "data", "validation"), transform=raw_transform)

    with torch.no_grad():
        for episode in range(n_episodes):
            support, query, support_labels, query_labels = create_episode(dataset, n_way, n_shot, n_query, device)
            if support is None:
                print(f"Episodio {episode + 1} omitido por datos insuficientes")
                continue

            scores = model(support, query)
            target_labels = torch.zeros(query.size(0), support.size(0), device=device)
            for i in range(query.size(0)):
                for j in range(support.size(0)):
                    if query_labels[i] == support_labels[j]:
                        target_labels[i, j] = 1.0
            scores_flat = scores.view(-1)
            target_flat = target_labels.view(-1)
            predictions = (torch.sigmoid(scores_flat) > 0.5).float()

            # Mapear predicciones binarias a clases
            episode_predictions = []
            for i in range(query.size(0)):
                query_scores = scores[i]
                max_score_idx = torch.argmax(query_scores).item()
                predicted_class = support_labels[max_score_idx].item()
                episode_predictions.append(predicted_class)
            all_predictions.extend(episode_predictions)
            all_labels.extend(query_labels.cpu().numpy())

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_predictions) if all_predictions else 0.0
    f1 = f1_score(all_labels, all_predictions, average="weighted") if all_predictions else 0.0
    conf_matrix = confusion_matrix(all_labels, all_predictions) if all_predictions else np.zeros((n_way, n_way))

    # Predicciones de muestra
    sample_indices = np.random.choice(len(raw_dataset), min(num_images, len(raw_dataset)), replace=False)
    sample_images = []
    sample_labels = []
    sample_paths = []
    for idx in sample_indices:
        img, label = raw_dataset[idx]
        img = img.unsqueeze(0).to(device)
        sample_images.append(img)
        sample_labels.append(label)
        sample_paths.append(raw_dataset.imgs[idx][0])

    sample_images = torch.cat(sample_images, dim=0) if sample_images else torch.tensor([])
    sample_predictions = []
    if sample_images.size(0) > 0:
        max_attempts = 3  # Reintentar hasta 3 veces si falla
        for _ in range(max_attempts):
            support, _, support_labels, _ = create_episode(dataset, n_way, n_shot, 0, device)
            if support is not None:
                scores = model(support, sample_images)
                for i in range(sample_images.size(0)):
                    query_scores = scores[i]
                    max_score_idx = torch.argmax(query_scores).item()
                    predicted_class = support_labels[max_score_idx].item()
                    sample_predictions.append(predicted_class)
                break
        else:
            print("No se pudo crear episodio para predicciones de muestra tras varios intentos")
            sample_predictions = [0] * len(sample_images)

    # Visualizar predicciones
    if sample_images.size(0) > 0:
        plt.figure(figsize=(15, 5))
        for i in range(len(sample_images)):
            plt.subplot(1, num_images, i+1)
            img = sample_images[i].cpu().permute(1, 2, 0).numpy()
            img = denormalize(torch.from_numpy(img).float()).numpy()
            plt.imshow(img)
            true_label = class_names[sample_labels[i]]
            pred_label = class_names[sample_predictions[i]] if sample_predictions else "Desconocido"
            is_correct = "Correcto" if true_label == pred_label else "Incorrecto"
            plt.title(f"Real: {true_label}\nPred: {pred_label}\n({is_correct})")
            plt.axis('off')
        plt.savefig(os.path.join(results_path, "relation_sample_predictions.png"))
        plt.close()

    # Guardar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names[:n_way], yticklabels=class_names[:n_way])
    plt.title("Matriz de Confusión - Relation Network")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.savefig(os.path.join(results_path, "confusion_matrix.png"))
    plt.close()

    # Guardar métricas y predicciones
    with open(os.path.join(results_path, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Modelo: Relation Network\n")
        f.write(f"n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, n_episodes: {n_episodes}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Episodios válidos: {len(all_predictions)//n_query}/{n_episodes}\n")

    with open(os.path.join(results_path, "sample_predictions.txt"), "w") as f:
        f.write(f"Predicciones de muestra para Relation Network:\n")
        for i in range(len(sample_images)):
            true_label = class_names[sample_labels[i]]
            pred_label = class_names[sample_predictions[i]] if sample_predictions else "Desconocido"
            is_correct = "Correcto" if true_label == pred_label else "Incorrecto"
            f.write(f"Imagen {i}: Real: {true_label}, Pred: {pred_label}, {is_correct}, Ruta: {sample_paths[i]}\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Resultados guardados en {results_path}")
    return accuracy, f1, conf_matrix

# Configuración principal
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cargar datos de validación
    BASE_DIR = r"C:\Users\lenovo\Desktop\SignLanguageRecognition"
    val_dataset = ImageFolder(os.path.join(BASE_DIR, "data", "validation"), transform=transform)

    # Configurar modelo
    model = RelationNetwork().to(device)
    model_path = os.path.join(BASE_DIR, "models", "few_shot", "relation_network_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modelo cargado desde {model_path}")

    # Definir ruta de resultados
    RESULTS_PATH = os.path.join(BASE_DIR, "results", "few_shot", "relation_network")
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Evaluar el modelo
    accuracy, f1, conf_matrix = evaluate_few_shot(model, val_dataset, RESULTS_PATH, n_way=3, n_shot=1, n_query=1, n_episodes=20, device=device, num_images=3)

if __name__ == "__main__":
    main()
