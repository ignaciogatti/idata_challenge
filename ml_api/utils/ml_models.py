import tensorflow as tf
import numpy as np

class CIFAR_Detectition_Pre_trained_Model():
    """Class to load a pre-trained TF model and run inference."""
    
    def __init__(self, path, input_size):
        
        self._model = self.load_model(path)
        self._input_size = input_size
        
    
    def preprocess_input(self, input_image):
        """Function to normalize the image to be consumed by the model.
        Args:
          img:  raw input image.

        Returns:
          image: image normalized.
        """
        image = tf.image.resize(input_image, self._input_size)
        image = image[tf.newaxis,...]
        return image
    
        
    
    def load_model(self, path):
        """Function to load a pre-trained model for cifar10 object recognition.
        Args:

        Returns:
          dog_breed_detection_model: TF pre-trained model.
        """
        model = tf.keras.models.load_model(path)
        return model
    
    
    def predict(self, img):
        """Function to predict the object from a single example
        Args:
          img:  raw input image.

        Returns:
          prediction: array with the probability assosiated to each class.
        """
                    
        tensor = self.preprocess_input(img)
        
        prediction = self._model.predict(tensor)
        
        return prediction
    
    
    def get_model(self):
        """Function to get the model

        Returns:
          model: TF model used by the object.
        """
        return self._model
    
    
    def compute_predictions(self, ds):
        """Function to get the model

        Returns:
          model: TF model used by the object.
        """
        
        #Prepare data
        size = self._input_size
        AUTOTUNE = tf.data.AUTOTUNE
        BATCH_SIZE= 32

        ds_normalized = ds.map(lambda x, y: (tf.image.resize(x, size, method=tf.image.ResizeMethod.AREA), y))
        ds_normalized = ds_normalized.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
        
        #Compute confusion matrix
        y_pred = self._model.predict(ds_normalized)
        y_test = np.concatenate([y for x, y in ds_normalized], axis=0)
        y_pred_classes = np.argmax(y_pred,axis = 1)
        
        return y_pred, y_test, y_pred_classes


  