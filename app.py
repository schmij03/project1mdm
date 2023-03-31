from flask import Flask, render_template, request
import onnxruntime
import numpy as np
from PIL import Image


app = Flask(__name__)

session = onnxruntime.InferenceSession('mnist-12-int8.onnx')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['image']
    img = Image.open(file.stream).convert('L') # convert to grayscale
    img = img.resize((28, 28)) # resize to 28x28
    img = np.array(img).reshape(1, 1, 28, 28).astype(np.float32) # reshape to match ONNX input format
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred = session.run([output_name], {input_name: img})[0]
    class_idx = np.argmax(pred)
    result = f'Predicted digit: {class_idx}'

    # Create a new HTML element with the predicted result and uploaded image
    html = f'<div class="alert alert-success" role="alert"> \
                <h4 class="alert-heading">Prediction result:</h4> \
                <p>{result}</p> \
            </div>'

    return html

if __name__ == '__main__':
    app.run(debug=True)
