from flask import Flask, request, send_file, render_template_string
import os
from PIL import Image
import torch
from io import BytesIO
from style_transfer import style_transfer_traced  # Импортируем функцию из вашего кода

app = Flask(__name__)

# Путь к трассированной модели
MODEL_PATH = 'style_transfer_10_2.ptl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.route('/')
def index():
    with open('templates/index.html', 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/upload', methods=['POST'])
def upload():
    # Получение загруженного изображения
    if 'content_image' not in request.files:
        return "No file uploaded", 400

    content_image = request.files['content_image']

    if content_image.filename == '':
        return "No selected file", 400

    # Сохранение временного файла
    input_path = "temp_content.jpg"
    content_image.save(input_path)

    try:
        # Выполнение переноса стиля
        result_image = style_transfer_traced(MODEL_PATH, None, input_path, DEVICE)

        # Сохранение результата в памяти
        output = BytesIO()
        result_pil = Image.fromarray(result_image)
        result_pil.save(output, format='JPEG')
        output.seek(0)

        # Удаление временного файла
        os.remove(input_path)

        return send_file(output, mimetype='image/jpeg')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
