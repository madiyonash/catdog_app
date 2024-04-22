from flask import Flask, request, render_template, send_file
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
model = tf.keras.models.load_model("model_alex.h5")  # Update to your model's path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Load the image and convert to RGB
            img = Image.open(file.stream).convert('RGB')
            # Resize the image to match the model's expected input
            img = img.resize((128, 128))
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize the pixel values
            img_array = img_array.reshape((1, 128, 128, 3))  # Reshape for model

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)

            # Determine title based on prediction
            title = "It's a Dog!!" if predicted_class[0] == 1 else "It's a Cat!!"
            
            # Display the image with prediction title
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')  # Turn off axis numbers and ticks

            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')

    return '''
    <!doctype html>
    <style>
    body {
        font-family: Georgia, serif;
        background-color:thistle;
        text-align:center;
    }
    </style>
    <title>Cats and Dogs Predictor</title>
    <h1>Cats and Dogs Predictor</h1>
    <p>Upload an image to classify it as a cat or dog! Please upload a PNG or JPEG.</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </body>
    '''

if __name__ == '__main__':
    app.run(debug=True)