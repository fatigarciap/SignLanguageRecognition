import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score
import time

# Definir el modelo MAML
class MAMLModel(nn.Module):
    def __init__(self):
        super(MAMLModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(512, 64)  # Ajustar salida

    def forward(self, x, params=None):
        if params is None:
            return self.backbone(x)
        # Aplicar parámetros adaptados (clonados sin gradientes en evaluación)
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].data.clone().detach()
        return self.backbone(x)

# Función para crear un episodio
def create_episode(dataset, n_way=3, n_shot=15, n_query=5, device="cpu"):
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
            continue
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

# Función para un paso de adaptación MAML (entrenamiento)
def maml_update_train(model, support, support_labels, inner_lr=0.005, inner_steps=2):
    params = {name: param.clone().detach().requires_grad_(True) for name, param in model.named_parameters()}
    optimizer = torch.optim.SGD([param for param in params.values()], lr=inner_lr)

    for _ in range(inner_steps):
        support_features = model(support)
        loss = nn.CrossEntropyLoss()(support_features, support_labels)
        grads = torch.autograd.grad(loss, params.values(), allow_unused=True)
        for (name, param), grad in zip(params.items(), grads):
            if grad is not None:
                param.data -= inner_lr * grad

    return params

# Función para adaptación en evaluación (sin gradientes)
def maml_update_eval(model, support, support_labels, inner_lr=0.005, inner_steps=2):
    params = {name: param.clone().detach() for name, param in model.named_parameters()}
    for _ in range(inner_steps):
        support_features = model(support, params=params)
        loss = nn.CrossEntropyLoss()(support_features, support_labels)
        # No calculamos gradientes, solo simulamos la actualización
        grads = torch.autograd.grad(loss, params.values(), allow_unused=True, create_graph=False)
        for (name, param), grad in zip(params.items(), grads):
            if grad is not None:
                params[name] = param - inner_lr * grad
    return params

# Función para entrenar
def train_maml(model, train_dataset, n_way=3, n_shot=15, n_query=5, n_episodes=50, device="cpu", inner_lr=0.005, inner_steps=2):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_accuracy = 0.0

    for episode in range(n_episodes):
        support, query, support_labels, query_labels = create_episode(train_dataset, n_way, n_shot, n_query, device)
        if support is None:
            continue

        # Adaptación interna
        adapted_params = maml_update_train(model, support, support_labels, inner_lr, inner_steps)
        # Evaluar con consulta usando parámetros adaptados
        query_features = model(query, params=adapted_params)
        _, predictions = torch.max(query_features, dim=1)

        accuracy = accuracy_score(query_labels.cpu().numpy(), predictions.cpu().numpy())
        total_accuracy += accuracy

        loss = nn.CrossEntropyLoss()(query_features, query_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episodio {episode + 1}/{n_episodes} - Precisión: {accuracy:.4f}")

    avg_accuracy = total_accuracy / n_episodes if n_episodes > 0 else 0.0
    return avg_accuracy

# Función para evaluar
def evaluate_maml(model, val_dataset, n_way=3, n_shot=15, n_query=5, n_episodes=20, device="cpu", inner_lr=0.005, inner_steps=2):
    model.eval()
    total_accuracy = 0.0

    with torch.no_grad():
        for episode in range(n_episodes):
            support, query, support_labels, query_labels = create_episode(val_dataset, n_way, n_shot, n_query, device)
            if support is None:
                continue
            # Adaptación interna sin gradientes
            adapted_params = maml_update_eval(model, support, support_labels, inner_lr, inner_steps)
            query_features = model(query, params=adapted_params)
            _, predictions = torch.max(query_features, dim=1)

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
    model = MAMLModel().to(device)

    # Entrenar el modelo
    print("Iniciando entrenamiento de MAML...")
    start_time = time.time()
    train_accuracy = train_maml(model, train_dataset, n_way=3, n_shot=15, n_query=5, n_episodes=50, device=device)
    print(f"Precisión promedio de entrenamiento: {train_accuracy:.4f}")
    print(f"Tiempo de entrenamiento: {time.time() - start_time:.2f} segundos")

    # Evaluar el modelo
    print("Evaluando el modelo...")
    start_time = time.time()
    val_accuracy = evaluate_maml(model, val_dataset, n_way=3, n_shot=15, n_query=5, n_episodes=20, device=device)
    print(f"Precisión promedio de validación: {val_accuracy:.4f}")
    print(f"Tiempo de evaluación: {time.time() - start_time:.2f} segundos")

    # Guardar el modelo
    os.makedirs("models/few_shot", exist_ok=True)
    model_path = "models/few_shot/maml_model_fixed_sequential.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado como {model_path}")

if __name__ == "__main__":
    main()