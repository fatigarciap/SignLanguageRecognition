import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score
import time
from torch.optim.lr_scheduler import StepLR

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
        )  # Red de relación

    def forward(self, support, query):
        batch_size = query.size(0)
        n_support = support.size(0)
        
        # Extraer características
        support_features = self.backbone(support).view(n_support, -1)
        query_features = self.backbone(query).view(batch_size, -1)
        
        # Crear pares (query, support)
        support_expanded = support_features.unsqueeze(0).repeat(batch_size, 1, 1)
        query_expanded = query_features.unsqueeze(1).repeat(1, n_support, 1)
        relations = torch.cat((support_expanded, query_expanded), dim=-1)
        
        # Pasar por la red de relación
        scores = self.fc(relations).view(batch_size, n_support)
        return scores

# Función para crear un episodio
def create_episode(dataset, n_way=3, n_shot=10, n_query=2, device="cpu"):
    class_names = dataset.classes
    all_indices = list(range(len(dataset)))
    selected_classes = np.random.choice(len(class_names), n_way, replace=False)
    
    support_set = []
    query_set = []
    support_labels = []
    query_labels = []

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

# Función para entrenar
def train_relation_network(model, train_dataset, n_way=3, n_shot=10, n_query=2, n_episodes=200, device="cpu"):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    total_accuracy = 0.0

    for episode in range(n_episodes):
        support, query, support_labels, query_labels = create_episode(train_dataset, n_way, n_shot, n_query, device)
        if support is None:
            print(f"Episodio {episode + 1} omitido por datos insuficientes")
            continue

        current_n_shot = support.size(0) // n_way
        current_n_query = query.size(0) // n_way
        
        # Calcular relaciones
        scores = model(support, query)
        
        # Crear etiquetas binarias para relaciones
        target_labels = torch.zeros(current_n_query * n_way, current_n_shot * n_way, device=device)
        for i in range(current_n_query):
            for j in range(current_n_shot):
                class_idx = query_labels[i * n_way // current_n_query]
                if support_labels[j * n_way // current_n_shot] == class_idx:
                    target_labels[i * n_way, j * n_way] = 1.0
        
        # Aplanar para la pérdida
        scores_flat = scores.view(-1)
        target_flat = target_labels.view(-1)
        
        # Calcular precisión
        predictions = (torch.sigmoid(scores_flat) > 0.5).float()
        accuracy = accuracy_score(target_flat.cpu().numpy(), predictions.cpu().numpy())
        total_accuracy += accuracy

        # Pérdida y optimización
        loss = criterion(scores_flat, target_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (episode + 1) % 10 == 0:
            print(f"Episodio {episode + 1}/{n_episodes} - Precisión: {accuracy:.4f} - Similitudes min/max: {scores_flat.min():.4f}/{scores_flat.max():.4f}")

    avg_accuracy = total_accuracy / n_episodes if n_episodes > 0 else 0.0
    return avg_accuracy

# Función para evaluar
def evaluate_relation_network(model, val_dataset, n_way=3, n_shot=10, n_query=2, n_episodes=20, device="cpu"):
    model.eval()
    total_accuracy = 0.0

    with torch.no_grad():
        for episode in range(n_episodes):
            support, query, support_labels, query_labels = create_episode(val_dataset, n_way, n_shot, n_query, device)
            if support is None:
                print(f"Episodio {episode + 1} omitido por datos insuficientes")
                continue

            current_n_shot = support.size(0) // n_way
            current_n_query = query.size(0) // n_way
            
            scores = model(support, query)
            
            # Crear etiquetas binarias
            target_labels = torch.zeros(current_n_query * n_way, current_n_shot * n_way, device=device)
            for i in range(current_n_query):
                for j in range(current_n_shot):
                    class_idx = query_labels[i * n_way // current_n_query]
                    if support_labels[j * n_way // current_n_shot] == class_idx:
                        target_labels[i * n_way, j * n_way] = 1.0
            
            scores_flat = scores.view(-1)
            target_flat = target_labels.view(-1)
            
            predictions = (torch.sigmoid(scores_flat) > 0.5).float()
            accuracy = accuracy_score(target_flat.cpu().numpy(), predictions.cpu().numpy())
            total_accuracy += accuracy
            print(f"Episodio {episode + 1}/{n_episodes} - Precisión: {accuracy:.4f} - Similitudes: {scores_flat.min():.4f}/{scores_flat.max():.4f}")

    avg_accuracy = total_accuracy / n_episodes
    return avg_accuracy

# Configuración principal
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(os.path.join("data", "train"), transform=train_transform)
    val_dataset = ImageFolder(os.path.join("data", "validation"), transform=val_transform)

    model = RelationNetwork().to(device)

    print("Iniciando entrenamiento de Relation Network...")
    start_time = time.time()
    train_accuracy = train_relation_network(model, train_dataset, n_way=3, n_shot=10, n_query=2, n_episodes=200, device=device)
    print(f"Precisión promedio de entrenamiento: {train_accuracy:.4f}")
    print(f"Tiempo de entrenamiento: {time.time() - start_time:.2f} segundos")

    print("Evaluando el modelo...")
    start_time = time.time()
    val_accuracy = evaluate_relation_network(model, val_dataset, n_way=3, n_shot=10, n_query=2, n_episodes=20, device=device)
    print(f"Precisión promedio de validación: {val_accuracy:.4f}")
    print(f"Tiempo de evaluación: {time.time() - start_time:.2f} segundos")

    os.makedirs("models/few_shot", exist_ok=True)
    model_path = "models/few_shot/relation_network_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado como {model_path}")

if __name__ == "__main__":
    main()