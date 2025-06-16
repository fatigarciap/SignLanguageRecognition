import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score
import time

# Definir el modelo Matching Network
class MatchingNetwork(nn.Module):
    def __init__(self):
        super(MatchingNetwork, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Congelar todas las capas inicialmente
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Descongelar layer4 y fc
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        self.backbone.fc = nn.Identity()  # Eliminar la capa fully connected
        self.fc = nn.Linear(512, 64)  # Reducir dimensionalidad
        self.attention = nn.Linear(64, 1)  # Capa de atención simple

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Aplana
        features = self.fc(features)
        return features

# Función para crear un episodio
def create_episode(dataset, n_way=3, n_shot=5, n_query=2, device="cpu"):
    class_names = dataset.classes
    all_indices = list(range(len(dataset)))
    selected_classes = np.random.choice(len(class_names), n_way, replace=False)
    
    support_set = []
    query_set = []
    support_labels = []
    query_labels = []

    # Contar imágenes por clase
    class_counts = np.bincount(dataset.targets)
    for i, class_idx in enumerate(selected_classes):
        if class_counts[class_idx] < n_shot + n_query:
            print(f"Error: Clase {class_names[class_idx]} tiene {class_counts[class_idx]} imágenes (< {n_shot + n_query})")
            return None, None, None, None
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
        return None, None, None, None
    support = torch.stack(support_set).to(device)
    query = torch.stack(query_set).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
    query_labels = torch.tensor(query_labels, dtype=torch.long).to(device)
    return support, query, support_labels, query_labels

# Función para calcular similitudes con atención
def compute_similarities(model, support, query):
    support_features = model(support)
    query_features = model(query)
    support_norm = support_features / support_features.norm(dim=1, keepdim=True)
    query_norm = query_features / query_features.norm(dim=1, keepdim=True)
    
    # Calcular atención sobre soporte
    attention_weights = torch.softmax(model.attention(support_features), dim=0)
    support_weighted = support_norm * attention_weights
    
    # Similitud coseno
    similarities = torch.matmul(query_norm, support_weighted.t())
    return similarities

# Función para entrenar
def train_matching_network(model, train_dataset, n_way=3, n_shot=5, n_query=2, n_episodes=100, device="cpu"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)  # Solo parámetros descongelados
    total_accuracy = 0.0

    for episode in range(n_episodes):
        support, query, support_labels, query_labels = create_episode(train_dataset, n_way, n_shot, n_query, device)
        if support is None:
            print(f"Episodio {episode + 1} omitido por datos insuficientes")
            continue

        # Calcular similitudes
        similarities = compute_similarities(model, support, query)
        # Predicciones como distribución softmax sobre similitudes
        scores = similarities
        _, predictions = torch.max(scores, dim=1)

        accuracy = accuracy_score(query_labels.cpu().numpy(), predictions.cpu().numpy())
        total_accuracy += accuracy

        loss = criterion(scores, query_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verificar gradientes
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        if (episode + 1) % 10 == 0:
            print(f"Episodio {episode + 1}/{n_episodes} - Precisión: {accuracy:.4f} - Similitudes min/max: {scores.min():.4f}/{scores.max():.4f} - Gradientes: {has_grad}")

    avg_accuracy = total_accuracy / n_episodes if n_episodes > 0 else 0.0
    return avg_accuracy

# Función para evaluar
def evaluate_matching_network(model, val_dataset, n_way=3, n_shot=5, n_query=2, n_episodes=20, device="cpu"):
    model.eval()
    total_accuracy = 0.0

    with torch.no_grad():
        for episode in range(n_episodes):
            support, query, support_labels, query_labels = create_episode(val_dataset, n_way, n_shot, n_query, device)
            if support is None:
                print(f"Episodio {episode + 1} omitido por datos insuficientes")
                continue

            # Calcular similitudes
            similarities = compute_similarities(model, support, query)
            # Predicciones como distribución softmax sobre similitudes
            scores = similarities
            _, predictions = torch.max(scores, dim=1)

            accuracy = accuracy_score(query_labels.cpu().numpy(), predictions.cpu().numpy())
            total_accuracy += accuracy

    avg_accuracy = total_accuracy / n_episodes
    return avg_accuracy

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

    # Cargar datos directamente (sin DataLoader)
    train_dataset = ImageFolder(os.path.join("data", "train"), transform=transform)
    val_dataset = ImageFolder(os.path.join("data", "validation"), transform=transform)

    # Configurar el modelo
    model = MatchingNetwork().to(device)

    # Entrenar el modelo
    print("Iniciando entrenamiento de Matching Network...")
    start_time = time.time()
    train_accuracy = train_matching_network(model, train_dataset, n_way=3, n_shot=5, n_query=2, n_episodes=100, device=device)
    print(f"Precisión promedio de entrenamiento: {train_accuracy:.4f}")
    print(f"Tiempo de entrenamiento: {time.time() - start_time:.2f} segundos")

    # Evaluar el modelo
    print("Evaluando el modelo...")
    start_time = time.time()
    val_accuracy = evaluate_matching_network(model, val_dataset, n_way=3, n_shot=5, n_query=2, n_episodes=20, device=device)
    print(f"Precisión promedio de validación: {val_accuracy:.4f}")
    print(f"Tiempo de evaluación: {time.time() - start_time:.2f} segundos")

    # Guardar el modelo
    os.makedirs("models/few_shot", exist_ok=True)
    model_path = "models/few_shot/matching_network_model_unfrozen.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado como {model_path}")

if __name__ == "__main__":
    main()