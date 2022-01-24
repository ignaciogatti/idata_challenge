import os
import tensorflow as tf
from flask import Flask
from .utils.ml_models import CIFAR_Detectition_Pre_trained_Model

app = Flask(__name__)

model_dir = os.path.join(os.getcwd(), 'ml_api/models/cifar_xception/')
cifar_detection_model = CIFAR_Detectition_Pre_trained_Model(model_dir, (96, 96))

cnn_model_dir = os.path.join(os.getcwd(), 'ml_api/models/cifar_cnn_augmentation/')
cifar_cnn_detection_model = CIFAR_Detectition_Pre_trained_Model(cnn_model_dir, (32, 32))

print('ML models loaded')

import ml_api.views