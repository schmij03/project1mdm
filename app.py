from flask import Flask, render_template, request
import torch
from PIL import Image
from torchvision import transforms


app = Flask(__name__)

model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)


def predict(image_path):
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
        return prediction
    

    
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        prediction = predict(file)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
