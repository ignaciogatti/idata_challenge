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
