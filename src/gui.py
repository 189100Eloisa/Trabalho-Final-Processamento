import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import models, transforms
import torch.nn as nn
from ultralytics import YOLO

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo de classificação
def load_classification_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Classificação binária: Cachorro e Gato
    model = model.to(device)

    # Carregar pesos do modelo treinado
    model_path = 'best_model.pth'  # Caminho do seu modelo de classificação treinado
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

classification_model = load_classification_model()

# Carregar o modelo de detecção YOLO
yolo_model = YOLO("yolov5su.pt")  # YOLO pré-treinado (model 'u' para melhor desempenho)

# Transformações para pré-processamento da imagem
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Mapeamento de classes
class_names = ['Cachorro', 'Gato']

# Função para prever a classe de uma imagem cortada
def predict_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classification_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Função para processar a imagem e exibir os resultados
def process_image(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        return None, "Erro ao carregar a imagem."

    # Detecção de objetos com YOLO
    results = yolo_model(image)

    # Copiar imagem para exibição com as caixas desenhadas
    annotated_image = image.copy()

    # Iterar pelos objetos detectados
    for result in results:  # Cada `result` contém os dados de um frame (imagem)
        boxes = result.boxes  # Caixas delimitadoras
        for box in boxes:  # Iterar por cada objeto detectado
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da caixa
            conf = box.conf[0]  # Confiança da detecção
            cls = int(box.cls[0])  # Classe YOLO (não usamos aqui para classificar diretamente)

            # Cortar o objeto detectado para classificação
            cropped_object = image[y1:y2, x1:x2]

            # Classificar o objeto detectado (Cachorro ou Gato)
            class_name = predict_image(cropped_object)

            # Definir a cor da caixa com base na classe detectada
            if class_name == 'Cachorro':
                color = (0, 0, 255)  # Vermelho para cachorro
            elif class_name == 'Gato':
                color = (255, 0, 0)  # Azul para gato
            else:
                color = (0, 255, 0)  # Verde para outros casos (se algum objeto não for cachorro ou gato)

            # Desenhar retângulo na imagem
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Mudar a posição do texto para aparecer acima da caixa delimitadora
            # Adicionar o texto logo acima da caixa, mas com uma pequena distância
            text_position = (x1, y1 - 10) if y1 > 30 else (x1, y1 + 20)  # Ajuste se a caixa estiver muito perto do topo
            cv2.putText(annotated_image, f"{class_name} ({conf:.2f})",
                        text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_image, None

# Função para redimensionar a imagem para um tamanho adequado
def resize_image_for_display(image, max_width=1200, max_height=800):
    # Obter as dimensões da imagem original
    height, width, _ = image.shape

    # Calcular a proporção de redimensionamento para manter a imagem proporcional
    ratio = min(max_width / width, max_height / height)

    # Calcular novas dimensões
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Redimensionar a imagem
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Função para selecionar e exibir imagem
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;*.png")])
    if not file_path:
        return  # O usuário cancelou a seleção

    # Processar a imagem
    processed_image, error = process_image(file_path)
    if error:
        messagebox.showerror("Erro", error)
        return

    # Redimensionar a imagem para caber na interface
    processed_image_resized = resize_image_for_display(processed_image)

    # Converter a imagem redimensionada para exibição
    processed_image_rgb = cv2.cvtColor(processed_image_resized, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(processed_image_rgb)

    # Exibir imagem no label
    photo = ImageTk.PhotoImage(pil_image)
    image_label.config(image=photo)
    image_label.image = photo

    # Exibir resultado de detecção na área de texto
    result_text = "Objetos detectados:\n"
    results = yolo_model(processed_image)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cropped_object = processed_image[y1:y2, x1:x2]
            class_name = predict_image(cropped_object)
            result_text += f"{class_name}: {conf:.2f}\n"

    result_label.config(text=result_text)

# Configuração da interface gráfica
root = tk.Tk()
root.title("Classificador de Cães e Gatos com Detecção")

# Frame com barra de rolagem
frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

# Barra de rolagem para a tela
canvas = tk.Canvas(frame)
canvas.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

canvas.configure(yscrollcommand=scrollbar.set)

image_label = tk.Label(canvas)
canvas.create_window((0, 0), window=image_label, anchor="nw")

# Botão para selecionar imagem
button = tk.Button(frame, text="Selecionar Imagem", command=select_image, bg="#4CAF50", fg="white", font=("Helvetica", 12))
button.pack(pady=10)

# Label para exibir resultados de detecção
result_label = tk.Label(frame, text="Resultado: ", font=("Helvetica", 12))
result_label.pack(pady=10)

root.mainloop()
