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
    model = models.resnet18(weights=None)  # Utilizar weights=None em vez de pretrained=False
    num_ftrs = model.fc.in_features
    # Adicionar Dropout para coincidir com o modelo treinado
    model.fc = nn.Sequential(
        nn.Dropout(0.6),  # Dropout de 50%
        nn.Linear(num_ftrs, 2)  # Classificação binária: Cachorro e Gato
    )
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
            text_position = (x1, y1 - 10) if y1 > 30 else (x1, y1 + 20)
            cv2.putText(annotated_image, f"{class_name} ({conf:.2f})",
                        text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_image, None

# Função para redimensionar a imagem para um tamanho adequado
def resize_image_for_display(image, max_width=800, max_height=600):
    # Obter as dimensões da imagem original
    height, width, _ = image.shape
    ratio = min(max_width / width, max_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Função para selecionar e exibir imagem
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;*.png")])
    if not file_path:
        return  # O usuário cancelou a seleção

    progress_label.config(text="Processando... Por favor, aguarde.")
    processed_image, error = process_image(file_path)
    if error:
        messagebox.showerror("Erro", error)
        return

    processed_image_resized = resize_image_for_display(processed_image)

    processed_image_rgb = cv2.cvtColor(processed_image_resized, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(processed_image_rgb)

    # Exibir imagem no label com transição suave (fade-in effect)
    photo = ImageTk.PhotoImage(pil_image)
    image_label.config(image=photo)
    image_label.image = photo
    image_label.after(500, lambda: image_label.config(image=photo))  # Fade-in effect

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
    progress_label.config(text="Processamento concluído.")
    
    # Exibir o botão de trocar imagem após carregar a imagem
    change_button.grid(row=1, column=0, pady=10, sticky="ew")  # Mostra o botão diretamente

# Função para trocar a imagem
def change_image():
    select_image()

# Configuração da interface gráfica com um design original e moderno
root = tk.Tk()
root.title("Classificador de Cães e Gatos com Detecção")
root.configure(bg="#F4F4F4")

# Criar o Canvas e a Scrollbar
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

# Configuração do Canvas
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Adicionar elementos dentro do scrollable_frame
frame = tk.Frame(scrollable_frame, padx=20, pady=20, bg="#F4F4F4")
frame.grid(row=0, column=0, sticky="nsew")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

image_label = tk.Label(frame, bg="#FFFFFF", bd=2, relief="solid")
image_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

button = tk.Button(frame, text="Selecionar Imagem", command=select_image, bg="#4CAF50", fg="white", font=("Helvetica", 14, "bold"), relief="flat", height=2)
button.grid(row=1, column=0, pady=10, sticky="ew")

# Inicialmente o botão de trocar imagem fica visível após a imagem ser carregada
change_button = tk.Button(frame, text="Trocar Imagem", command=change_image, bg="#FF9800", fg="white", font=("Helvetica", 14, "bold"), relief="flat", height=2)

progress_label = tk.Label(frame, text="", font=("Helvetica", 12), fg="#777777", bg="#F4F4F4")
progress_label.grid(row=2, column=0, columnspan=2, pady=10)

result_label = tk.Label(frame, text="Resultado: ", font=("Helvetica", 12), fg="#333333", bg="#F4F4F4", justify="left")
result_label.grid(row=3, column=0, columnspan=2, pady=10, sticky="w")

root.mainloop()
