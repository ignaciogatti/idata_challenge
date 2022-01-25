from ml_api import app, cifar_detection_model, cifar_cnn_detection_model

from flask import request, jsonify, render_template
from werkzeug.utils import secure_filename
from .utils.prediction_function import make_prediction

import os



UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def hello():
    '''
    Endpoint to access from browser
    Args:
    Return:
      html page
    '''

    return render_template('hello.html')


@app.route('/predictfront', methods=['POST'])
def predict_with_front():
    '''
    Endpoint that take an image as input and try to detect which object it is
    Args:
      file: a valid image file
    Return:
      html page
    '''


    # check if the post request has the file part 
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file'
        })
    
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({
            'error': 'Empty file'
        })
    
    
    if file and allowed_file(file.filename):
        #Upload image
        filename = secure_filename(file.filename)
        path = os.path.join(os.getcwd(), 'ml_api/download_images/' + filename)
        file.save(path)

        #make a prediction
        class_name, prob = make_prediction(path, cifar_cnn_detection_model)

        #Delete tmp image file
        if os.path.isfile(path):
            os.remove(path)

        #Everything works fine
        text = 'We detect a ' + class_name + ' in your image with prob ' + str(prob)
        return render_template('hello.html', prediction_text=text)

    return jsonify({
            'error': 'Something goes wrong'
        })



@app.route('/predict', methods=['POST'])
def predict():
    '''
    Endpoint that take an image as input and try to detect which object it is
    Args:
      file: a valid image file
    Return:
      JSON with class name and prob
    '''


    # check if the post request has the file part 
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file'
        })
    
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({
            'error': 'Empty file'
        })
    
    
    if file and allowed_file(file.filename):
        #Upload image
        filename = secure_filename(file.filename)
        path = os.path.join(os.getcwd(), 'ml_api/download_images/' + filename)
        file.save(path)

        #make a prediction
        class_name, prob = make_prediction(path, cifar_cnn_detection_model)

        #Delete tmp image file
        if os.path.isfile(path):
            os.remove(path)

        #Everything works fine
        return jsonify({
            'className' : class_name,
            'prob': str(prob)
        })

    return jsonify({
            'error': 'Something goes wrong'
        })