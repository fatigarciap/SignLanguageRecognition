
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Definir modelos few-shot
class RelationNetwork(nn.Module):
    def __init__(self, n_way):
        super(RelationNetwork, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, n_way)
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

class PrototypicalNetwork(nn.Module):
    def __init__(self, n_way):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, n_way)

    def forward(self, support, query):
        support_features = self.backbone(support).view(support.size(0), -1)
        query_features = self.backbone(query).view(query.size(0), -1)
        prototypes = support_features.mean(dim=0, keepdim=True)
        distances = torch.cdist(query_features, prototypes)
        return -distances

# Función para crear un episodio
def create_episode(dataset, n_way=2, n_shot=5, n_query=5, device="cpu"):
    all_indices = list(range(len(dataset)))
    class_counts = np.bincount(dataset.targets)
    valid_classes = [i for i, count in enumerate(class_counts) if count >= n_shot + n_query]
    if len(valid_classes) < n_way:
        print(f"Error: Solo {len(valid_classes)} clases tienen suficientes imágenes (>= {n_shot + n_query})")
        return None, None, None, None
    selected_classes = np.random.choice(valid_classes, n_way, replace=False)
    support_set, query_set, support_labels, query_labels = [], [], [], []
    for i, class_idx in enumerate(selected_classes):
        class_indices = [idx for idx in all_indices if dataset.targets[idx] == class_idx]
        np.random.shuffle(class_indices)
        support_indices = class_indices[:n_shot]
        query_indices = class_indices[n_shot:n_shot + n_query]
        for idx in support_indices + query_indices:
            img, _ = dataset[idx]
            (support_set if idx in support_indices else query_set).append(img)
        support_labels.extend([i] * n_shot)
        query_labels.extend([i] * n_query)
    return (torch.stack(support_set).to(device), torch.stack(query_set).to(device),
            torch.tensor(support_labels, dtype=torch.long).to(device),
            torch.tensor(query_labels, dtype=torch.long).to(device))

# Función para evaluar un modelo few-shot
def evaluate_fewshot(model, dataset, n_way=2, n_shot=5, n_query=5, n_episodes=20, device="cpu"):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            support, query, support_labels, query_labels = create_episode(dataset, n_way, n_shot, n_query, device)
            if support is None:
                continue
            scores = model(support, query)
            predictions = torch.argmax(scores, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(query_labels.cpu().numpy())
    return accuracy_score(all_labels, all_predictions) if all_predictions else 0.0

# Configuración principal
def main():
    device = torch.device("cpu")  # Fallback a CPU por ahora
    print(f"Usando dispositivo: {device}")

    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cargar datos
    BASE_DIR = r"C:\Users\lenovo\Desktop\SignLanguageRecognition"
    try:
        dataset = ImageFolder(os.path.join(BASE_DIR, "data", "train"), transform=transform)
        n_way = min(2, len(dataset.classes))  # Ajustar n_way a 2 o menos
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return

    # Lista de modelos few-shot
    fewshot_models = {
        "Relation Network": os.path.join(BASE_DIR, "models", "few_shot", "relation_network_model.pth"),
        "Prototypical Network": os.path.join(BASE_DIR, "models", "few_shot", "prototypical_model.pth")
    }  # Limitado a los que tienen definiciones compatibles por ahora

    # Evaluar modelos
    accuracies = {}
    for name, path in fewshot_models.items():
        if not os.path.exists(path):
            print(f"Modelo no encontrado: {name} ({path})")
            continue
        try:
            if "Relation" in name:
                model = RelationNetwork(n_way)
            elif "Prototypical" in name:
                model = PrototypicalNetwork(n_way)
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            accuracy = evaluate_fewshot(model, dataset, n_way=n_way, n_shot=5, n_query=5, n_episodes=20, device=device)
            accuracies[name] = accuracy
            print(f"{name}: Accuracy = {accuracy:.4f}")
        except Exception as e:
            print(f"Error cargando o evaluando {name}: {e}")

    # Generar gráfico de precisión
    if accuracies:
        plt.figure(figsize=(10, 6))
        plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
        plt.xlabel('Modelos Few-Shot')
        plt.ylabel('Precisión')
        plt.title('Comparación de Precisión de Modelos Few-Shot')
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies.values()):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
        plt.savefig(os.path.join(BASE_DIR, 'accuracy_comparison.png'))
        plt.show()
    else:
        print("No se pudieron evaluar modelos few-shot.")

if __name__ == "__main__":
    main()
