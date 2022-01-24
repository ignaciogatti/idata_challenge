from PIL import Image
import numpy as np


from ml_api import cifar_detection_model, cifar_cnn_detection_model

def make_prediction(path):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    pil_image = Image.open(path).convert("RGB")
    pix_image = np.asarray(pil_image)

    print('Making prediction...')

    #prob_prediction = cifar_detection_model.predict(pix_image)
    prob_prediction = cifar_cnn_detection_model.predict(pix_image)
    class_label = prob_prediction.argmax()

    print('Prediction Finished')


    return class_names[class_label], prob_prediction.reshape((-1))[class_label]   