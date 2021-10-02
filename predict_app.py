import base64
import numpy as np
import io
from PIL import Image
from tensorflow import keras
import tensorflow_addons as tfa
import csv
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('predict.html')

def get_model():
    global model
    model = load_model(r"model\final_model")
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    # normalize input
    image /= 255.0
    image = np.expand_dims(image, axis=0)

    return image

def load_classes(filename=r"data\class_names.csv"):
    global list_class_names
    
    with open(filename, newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        list_class_names = next(csv_reader)
    


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image'].split(",")[1]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(256, 256))
    
    prediction = np.squeeze(model.predict(processed_image))

    winning_class_index = np.argmax(prediction)
    winning_class = list_class_names[winning_class_index]
    confidence = prediction[winning_class_index].astype(float)

    response = {
        'prediction': {
            'winning_class': winning_class,
            'confidence': confidence
        }
    }
    return jsonify(response)


if __name__ == "__main__":
    # load class names first
    load_classes()

    # load trained tensorflow model
    print(" * Loading Keras model...")
    get_model()

    # start flask app
    app.run(port='8088',threaded=False)    