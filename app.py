from flask import Flask, render_template, request
import torch
from PIL import Image
from torchvision import transforms
import os

app = Flask(__name__)

model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)

def predict(image_path):
    try:
        # image preprocessing
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            class_index = predicted.cpu().numpy()[0]
            with open('imagenet_classes.txt') as f:
                classes = [line.strip() for line in f.readlines()]
                prediction = classes[class_index]
                prediction = prediction.split(",")[1].strip().capitalize()  # Get the class name and capitalize the first letter
            return prediction
    except Exception as e:
        error_msg = "Error: Failed to preprocess or classify the image. Please ensure the image is in a supported format and try again."
        print(f"{error_msg}\n{e}")
        return error_msg




@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    uploaded_image = None
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('static', filename)
        file.save(file_path)
        uploaded_image = file_path
        prediction = predict(file_path)
    if request.args.get('clear') == 'True':
        uploaded_image = None
        prediction = None
    return render_template('index.html', prediction=prediction, image=uploaded_image)


if __name__ == '__main__':
    app.run(debug=True)
