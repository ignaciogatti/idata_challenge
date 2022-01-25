# idata challenge

This repository implements a couple of DNN models to solve the well-known [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The models was developed using [Tensorflow](https://www.tensorflow.org/)

Basically, in this case I used two models:

- A custom CNN model trained from scratch
- A pre-trained [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16) model. In this case, only the head was trained in order to fit with the classes of the dataset.


Also, I developed a REST API to expose the model. For this purpose I used [Flask](https://flask.palletsprojects.com/en/2.0.x/).

You can acces to the model [here](https://drive.google.com/drive/folders/175HlLGUiLHWZw8YcJofASzNajTS3id6g?usp=sharing)


# Models evaluation

### Acuracy

- CNN model: 84%
- VGG16: 77%

### Confusion matrix

![Consfusion matrix](https://github.com/ignaciogatti/idata_challenge/blob/main/images/confusion_matrix.png)

From the confusion matrix, it is possible to observe, on one hand, that the "cat" is the hardest class to predict for both models; and on the hand, for each model the rate of true positive is quite similar around all classes. 
