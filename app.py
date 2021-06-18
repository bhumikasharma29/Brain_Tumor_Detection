from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np


# Keras
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import waitress
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'brain_tumor_model.h5'

session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    tf.keras.backend.set_session(session)
    # Load your trained model
    model = tf.keras.models.load_model(MODEL_PATH)

#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(240, 240))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    result = model.predict(x)
    
    if result <= 0.5:
        result = "Person does not have Brain Tumor"
    else:
        result = "Person has Brain Tumor"
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        with session.graph.as_default():
            tf.keras.backend.set_session(session)
            preds = model_predict(file_path, model)
        return preds
    return None


if __name__ == '__main__':
    app.debug = False
    port = int(os.environ.get('PORT', 33507))
    waitress.serve(app, port = port)

#app.run(debug=True)
