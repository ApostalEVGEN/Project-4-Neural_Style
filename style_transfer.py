import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Размер для изменения изображений
IMG_DIMENSIONS = (256, 256)

def load_traced_model(model_path, device):
    """
    Загружает трассированную модель для переноса стиля.
    :param model_path: Путь к трассированной модели.
    :param device: Устройство для выполнения вычислений (CPU/GPU).
    :return: Загруженная трассированная модель.
    """
    model = torch.jit.load(model_path, map_location=device)
    model.to(device).eval()
    return model

def preprocess_image(image_path):
    """
    Преобразует изображение в формат, подходящий для модели.
    :param image_path: Путь к изображению.
    :return: Подготовленное изображение в формате Torch Tensor.
    """
    img = Image.open(image_path).resize(IMG_DIMENSIONS)
    img = np.asarray(img).transpose(2, 0, 1)[0:3]
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img

def postprocess_image(output_tensor):
    """
    Преобразует выход модели в изображение.
    :param output_tensor: Выход модели в формате Torch Tensor.
    :return: Изображение в формате numpy.
    """
    output_image = output_tensor.squeeze().detach().cpu().numpy()
    output_image = np.clip(output_image, 0, 255).astype('uint8')
    output_image = output_image.transpose(1, 2, 0) 
    return output_image

def style_transfer_traced(model_path, image_path1, image_path2, device='cpu'):
    """
    Выполняет перенос стиля с использованием трассированной модели.
    :param model_path: Путь к трассированной модели.
    :param image_path1: Путь к изображению стиля (не используется в трассированной модели).
    :param image_path2: Путь к обыденному изображению.
    :param device: Устройство для выполнения (CPU/GPU).
    :return: Обработанное изображение.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = load_traced_model(model_path, device)
    content_img = preprocess_image(image_path2).to(device)

    output_tensor = model(content_img)

    output_image = postprocess_image(output_tensor)
    return output_image

