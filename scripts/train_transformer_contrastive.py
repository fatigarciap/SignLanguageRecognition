import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score

# Definir el modelo Transformer-based Contrastive
class TransformerContrastiveModel(nn.Module):
    def __init__(self):
        super(TransformerContrastiveModel, self).__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()  # Quitar la cabeza de clasificación
        self.fc = nn.Linear(768, 64)  # Ajustar salida del ViT

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Función para crear un episodio
def create_episode(dataset, n_way=3, n_shot=10, n_query=50, device="cpu"):
    class_names = dataset.classes
    all_indices = list(range(len(dataset)))
    selected_classes = np.random.choice(len(class_names), n_way, replace=False)
    
    support_set = []
    query_set = []
    support_labels = []
    query_labels = []

    for i, class_idx in enumerate(selected_classes):
        class_indices = [idx for idx in all_indices if dataset.targets[idx] == class_idx]
        if len(class_indices) < n_shot + n_query:
            continue  # Saltar si no hay suficientes imágenes
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
        return None, None, None, None  # Evitar episodios vacíos
    support = torch.stack(support_set).to(device)
    query = torch.stack(query_set).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
    query_labels = torch.tensor(query_labels, dtype=torch.long).to(device)
    return support, query, support_labels, query_labels

# Función para entrenar
def train_transformer_contrastive(model, train_dataset, n_way=3, n_shot=10, n_query=5, n_episodes=50, device="cpu"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_accuracy = 0.0

    for episode in range(n_episodes):
        support, query, support_labels, query_labels = create_episode(train_dataset, n_way, n_shot, n_query, device)
        if support is None:
            continue  # Saltar episodio inválido

        # Características de soporte y consulta
        support_features = model(support)
        query_features = model(query)
        # Usar similitud coseno como métrica de contraste
        distances = 1 - torch.nn.functional.cosine_similarity(query_features.unsqueeze(1), support_features.unsqueeze(0))
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

# Función para evaluar (para monitoreo durante entrenamiento)
def evaluate_transformer_contrastive(model, val_dataset, n_way=3, n_shot=10, n_query=5, n_episodes=50, device="cpu"):
    model.eval()
    total_accuracy = 0.0

    with torch.no_grad():
        for _ in range(n_episodes):
            support, query, support_labels, query_labels = create_episode(val_dataset, n_way, n_shot, n_query, device)
            if support is None:
                continue
            support_features = model(support)
            query_features = model(query)
            distances = 1 - torch.nn.functional.cosine_similarity(query_features.unsqueeze(1), support_features.unsqueeze(0))
            scores = -distances
            _, predictions = torch.max(scores, dim=1)

            accuracy = accuracy_score(query_labels.cpu().numpy(), predictions.cpu().numpy())
            total_accuracy += accuracy

    avg_accuracy = total_accuracy / n_episodes
    return avg_accuracy

# Configuración principal
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(os.path.join("data", "train"), transform=transform)
    val_dataset = ImageFolder(os.path.join("data", "validation"), transform=transform)

    model = TransformerContrastiveModel().to(device)

    print("Iniciando entrenamiento de Transformer-based Contrastive...")
    train_accuracy = train_transformer_contrastive(model, train_dataset, n_way=3, n_shot=10, n_query=5, n_episodes=50, device=device)
    print(f"Precisión promedio de entrenamiento: {train_accuracy:.4f}")

    print("Evaluando el modelo...")
    val_accuracy = evaluate_transformer_contrastive(model, val_dataset, n_way=3, n_shot=10, n_query=5, n_episodes=10, device=device)
    print(f"Precisión promedio de validación: {val_accuracy:.4f}")

    os.makedirs("models/few_shot", exist_ok=True)
    model_path = "models/few_shot/transformer_contrastive.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado como {model_path}")

if __name__ == "__main__":
    main()