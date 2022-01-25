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


### Precision/Recall report

#### Xception model

|              | precision          | recall             | f1-score           | support            |
|--------------|--------------------|--------------------|--------------------|--------------------|
| airplane     | 0.811214953271028  | 0.7469879518072289 | 0.7777777777777777 | 581.0              |
| automobile   | 0.8443017656500803 | 0.8885135135135135 | 0.8658436213991769 | 592.0              |
| bird         | 0.8123861566484517 | 0.707936507936508  | 0.7565733672603902 | 630.0              |
| cat          | 0.6552962298025135 | 0.6218057921635435 | 0.6381118881118881 | 587.0              |
| deer         | 0.6748554913294798 | 0.7460063897763578 | 0.708649468892261  | 626.0              |
| dog          | 0.7091757387247278 | 0.7768313458262351 | 0.7414634146341463 | 587.0              |
| frog         | 0.7124463519313304 | 0.8272425249169435 | 0.7655649500384321 | 602.0              |
| horse        | 0.8415841584158416 | 0.7130872483221476 | 0.7720254314259763 | 596.0              |
| ship         | 0.8508196721311475 | 0.8781725888324873 | 0.8642797668609492 | 591.0              |
| truck        | 0.8534923339011925 | 0.8240131578947368 | 0.8384937238493724 | 608.0              |
| accuracy     | 0.7728333333333334 | 0.7728333333333334 | 0.7728333333333334 | 0.7728333333333334 |
| macro avg    | 0.7765572851805793 | 0.7730597020989702 | 0.7728783410250369 | 6000.0             |
| weighted avg | 0.7764308370702534 | 0.7728333333333334 | 0.7726875979563305 | 6000.0             |


#### CNN model

|              | precision          | recall             | f1-score           | support |
|--------------|--------------------|--------------------|--------------------|---------|
| airplane     | 0.8511705685618729 | 0.8760757314974182 | 0.8634435962680237 | 581.0   |
| automobile   | 0.9570661896243292 | 0.9037162162162162 | 0.9296264118158123 | 592.0   |
| bird         | 0.8601134215500945 | 0.7222222222222222 | 0.7851596203623813 | 630.0   |
| cat          | 0.7145390070921985 | 0.686541737649063  | 0.7002606429192008 | 587.0   |
| deer         | 0.802710843373494  | 0.8514376996805112 | 0.8263565891472869 | 626.0   |
| dog          | 0.748829953198128  | 0.817717206132879  | 0.7817589576547231 | 587.0   |
| frog         | 0.8131386861313868 | 0.925249169435216  | 0.8655788655788655 | 602.0   |
| horse        | 0.9259259259259259 | 0.8389261744966443 | 0.8802816901408451 | 596.0   |
| ship         | 0.9087837837837838 | 0.9103214890016921 | 0.9095519864750633 | 591.0   |
| truck        | 0.9012738853503185 | 0.930921052631579  | 0.9158576051779935 | 608.0   |
| accuracy     | 0.846              | 0.846              | 0.846              | 0.846   |
| macro avg    | 0.8483552264591532 | 0.8463128698963441 | 0.8457875965540195 | 6000.0  |
| weighted avg | 0.8484843978704993 | 0.846              | 0.8456679781259134 | 6000.0  |

Looking both tables, it is possible to classify the classes in two groups: hard-predicted classes and well-predicted classes. The first group contains the classes with low precision score (>0.8), and the other one contains classes wit better precision score (<80).

In the case of the Xception model, the hard-predicted set include cat, deer, dog, frog; while the other classes are part of the well-predicted group. While in the CNN model, the hard-predicted classes set include cat and dog.
