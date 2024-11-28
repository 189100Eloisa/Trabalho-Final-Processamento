import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from torch.optim import lr_scheduler
import copy
import logging
import matplotlib.pyplot as plt  

# Configuração do log
LOG_FILE = "training_log.txt"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  
                        logging.FileHandler(LOG_FILE) 
                    ])

# Definição do número de épocas
NUM_EPOCHS = 10  

# Transformações do PyTorch com Data Augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Dataset personalizado com OpenCV para pré-processamento
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx or self._find_classes()

        self.image_paths = []
        self.labels = []

        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    if os.path.isfile(file_path):
                        self.image_paths.append(file_path)
                        self.labels.append(idx)

    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        classes.sort()
        return {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img_path = os.path.normpath(img_path)  
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Erro ao carregar imagem {img_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            if self.transform:
                image = self.transform(pil_image)

            return image, label
        except Exception as e:
            logging.error(f"Erro ao processar a imagem: {img_path}. Erro: {e}")
            blank_image = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(blank_image)
            try:
                font = ImageFont.load_default()
            except:
                font = None
            draw.text((75, 100), "?", fill='black', font=font)
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, -1  

# Função personalizada para lidar com erros no DataLoader
def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None] 
    return torch.utils.data.default_collate(batch)

# Configurações de treino e validação
train_dir = 'dataset/train'
val_dir = 'dataset/val'

train_dataset = CustomImageDataset(train_dir, transform=data_transforms['train'])
val_dataset = CustomImageDataset(val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
num_ftrs = model.fc.in_features

# Adicionar Dropout ao modelo
model.fc = nn.Sequential(
    nn.Dropout(0.5),  
    nn.Linear(num_ftrs, 2)  
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Função de treinamento
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch + 1}/{num_epochs}')
        logging.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

                if i % 10 == 0 or i == len(loader) - 1:
                    logging.info(f'{phase} {i + 1}/{len(loader)} Loss: {loss.item():.4f}')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_accuracies.append(epoch_acc)
                train_losses.append(epoch_loss)
            else:
                val_accuracies.append(epoch_acc)
                val_losses.append(epoch_loss)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    logging.info(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)

    # Gráfico de acurácia e perda
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(14, 5))

    # Subplot para Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label='Acurácia de Treinamento')
    plt.plot(epochs_range, val_accuracies, label='Acurácia de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.ylim(0, 1)
    plt.title('Acurácia de Treinamento e Validação por Época')
    plt.legend()
    plt.grid()

    # Subplot para Perda
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, label='Perda de Treinamento')
    plt.plot(epochs_range, val_losses, label='Perda de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Perda de Treinamento e Validação por Época')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('training_validation_metrics.png')
    plt.show()

    return model

# Iniciar treinamento
if __name__ == "__main__":
    model = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=NUM_EPOCHS)
    torch.save(model.state_dict(), 'best_model.pth')

# Ajustar o carregamento do modelo para a interface gráfica
def load_classification_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  
        nn.Linear(num_ftrs, 2) 
    )
    model = model.to(device)

    # Carregar pesos do modelo treinado
    model_path = 'best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
