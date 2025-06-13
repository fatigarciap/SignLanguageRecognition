import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score

# Definir la clase Prototypical Network
class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, 64)  # Ajustar según la salida del backbone

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Aplana las características
        x = self.fc(x)
        return x

# Función para calcular prototipos
def compute_prototypes(model, support, labels, n_way, n_shot):
    prototypes = []
    for i in range(n_way):
        class_indices = (labels == i).nonzero(as_tuple=True)[0]
        class_support = support[class_indices]
        class_features = model(class_support)
        prototype = class_features.mean(dim=0, keepdim=True)
        prototypes.append(prototype)
    return torch.cat(prototypes, dim=0)

# Función para crear un episodio
def create_episode(dataset, n_way=5, n_shot=5, n_query=5, device="cpu"):
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

# Función para entrenar
def train_model(model, train_dataset, n_way=5, n_shot=5, n_query=5, n_episodes=100, device="cpu"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_accuracy = 0.0

    for episode in range(n_episodes):
        support, query, support_labels, query_labels = create_episode(train_dataset, n_way, n_shot, n_query, device)
        prototypes = compute_prototypes(model, support, support_labels, n_way, n_shot)
        query_features = model(query)
        distances = torch.cdist(query_features, prototypes)
        scores = -distances
        _, predictions = torch.max(scores, dim=1)

        accuracy = accuracy_score(query_labels.cpu().numpy(), predictions.cpu().numpy())
        total_accuracy += accuracy

        loss = criterion(scores, query_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episodio {episode + 1}/{n_episodes} - Precisión: {accuracy:.4f}")

    avg_accuracy = total_accuracy / n_episodes if n_episodes > 0 else 0.0
    return avg_accuracy

# Función para evaluar
def evaluate_model(model, val_dataset, n_way=5, n_shot=5, n_query=5, n_episodes=10, device="cpu"):
    model.eval()
    total_accuracy = 0.0

    with torch.no_grad():
        for _ in range(n_episodes):
            support, query, support_labels, query_labels = create_episode(val_dataset, n_way, n_shot, n_query, device)
            prototypes = compute_prototypes(model, support, support_labels, n_way, n_shot)
            query_features = model(query)
            distances = torch.cdist(query_features, prototypes)
            scores = -distances
            _, predictions = torch.max(scores, dim=1)

            accuracy = accuracy_score(query_labels.cpu().numpy(), predictions.cpu().numpy())
            total_accuracy += accuracy

    avg_accuracy = total_accuracy / n_episodes
    return avg_accuracy

# Configuración principal
def main():
    # Detectar dispositivo (CPU o GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transformaciones para las imágenes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cargar datos
    train_dataset = ImageFolder(os.path.join("data", "train"), transform=transform)
    val_dataset = ImageFolder(os.path.join("data", "validation"), transform=transform)

    # Configurar el modelo
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Usar 'weights' en lugar de 'pretrained'
    backbone.fc = nn.Identity()  # Quitar la capa final
    model = PrototypicalNetwork(backbone).to(device)

    # Entrenar el modelo
    print("Iniciando entrenamiento...")
    train_accuracy = train_model(model, train_dataset, n_way=5, n_shot=5, n_query=5, n_episodes=100, device=device)
    print(f"Precisión promedio de entrenamiento: {train_accuracy:.4f}")

    # Evaluar el modelo
    print("Evaluando el modelo...")
    val_accuracy = evaluate_model(model, val_dataset, n_way=5, n_shot=5, n_query=5, n_episodes=10, device=device)
    print(f"Precisión promedio de validación: {val_accuracy:.4f}")

    # Guardar el modelo
    os.makedirs("models/few_shot", exist_ok=True)
    model_path = "models/few_shot/prototypical_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado como {model_path}")

if __name__ == "__main__":
    main()