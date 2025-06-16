import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
import time

# Configurar directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "few_shot", "relation_network_model.pth")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
NEW_GESTURE_DIR = os.path.join(DATA_DIR, "new_gestures")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Transformaciones para imágenes
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Definir modelo Relation Network
class RelationNetwork(torch.nn.Module):
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False  # Congelar backbone para inferencia
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

# Cargar modelo
model = RelationNetwork()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Cargar conjunto de soporte desde data/train (10 imágenes aleatorias)
support_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
support_indices = np.random.choice(len(support_dataset), 10, replace=False)
support_tensor = torch.stack([support_dataset[i][0] for i in support_indices]).to(device)

# Captura de video
cap = cv2.VideoCapture(0)
gesture_count = 0
new_gesture_name = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar manos con MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, c = frame.shape
            cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
            hand_region = frame[max(0, cy-100):min(h, cy+100), max(0, cx-100):min(w, cx+100)]
            if hand_region.size == 0:
                continue

            # Preprocesar y predecir
            query_tensor = transform(hand_region).unsqueeze(0).to(device)
            with torch.no_grad():
                scores = model(support_tensor, query_tensor)
                prediction = torch.sigmoid(scores).max().item()  # Valor de similitud máxima
            cv2.putText(frame, f"Pred: {prediction:.4f}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mostrar frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Capturar foto al pulsar barra espaciadora
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if new_gesture_name is None:
            new_gesture_name = input("Nombre del nuevo gesto: ")
            os.makedirs(os.path.join(NEW_GESTURE_DIR, new_gesture_name), exist_ok=True)
        gesture_count += 1
        img_path = os.path.join(NEW_GESTURE_DIR, new_gesture_name, f"gesture_{gesture_count}.jpg")
        cv2.imwrite(img_path, hand_region)
        print(f"Guardada imagen: {img_path}")

    # Salir con 'q'
    if key == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

if gesture_count > 0:
    print("Añade lógica para ajustar el modelo con las nuevas imágenes en", NEW_GESTURE_DIR)