# idata challenge

This repository implements a couple of DNN models to solve the well-known [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The models was developed using [Tensorflow](https://www.tensorflow.org/)

Basically, in this case I used two models:

- A custom CNN model trained from scratch
- A pre-trained [Xception](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception) model. In this case, only the head was trained in order to fit with the classes of the dataset.


Also, I developed a REST API to expose the model. For this purpose I used [Flask](https://flask.palletsprojects.com/en/2.0.x/).

You can acces to the model [here](https://drive.google.com/drive/folders/175HlLGUiLHWZw8YcJofASzNajTS3id6g?usp=sharing)


# Models evaluation

### Acuracy

- CNN model: 85%
- Xception (without fine-tuning): 77%
- Xception (with fine-tuning): 90%

### Confusion matrix

![Consfusion matrix](https://github.com/ignaciogatti/idata_challenge/blob/main/images/confusion_matrix.png)

From the confusion matrix, it is possible to observe, on one hand, that the "cat" is the hardest class to predict for both models; and on the hand, for each model the rate of true positive is quite similar around all classes. 


### Precision/Recall report

#### Xception model (without fine-tuning)

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


#### Xception (with fine-tuning)

|              | precision          | recall             | f1-score           | support |
|--------------|--------------------|--------------------|--------------------|---------|
| airplane     | 0.9131175468483816 | 0.9225473321858864 | 0.9178082191780822 | 581.0   |
| automobile   | 0.9425675675675675 | 0.9425675675675675 | 0.9425675675675675 | 592.0   |
| bird         | 0.9194078947368421 | 0.8873015873015873 | 0.9030694668820678 | 630.0   |
| cat          | 0.8460111317254174 | 0.7768313458262351 | 0.8099467140319716 | 587.0   |
| deer         | 0.8624420401854714 | 0.8913738019169329 | 0.8766692851531814 | 626.0   |
| dog          | 0.8062015503875969 | 0.8858603066439523 | 0.8441558441558442 | 587.0   |
| frog         | 0.9370748299319728 | 0.915282392026578  | 0.9260504201680673 | 602.0   |
| horse        | 0.9066666666666666 | 0.912751677852349  | 0.9096989966555183 | 596.0   |
| ship         | 0.9398998330550918 | 0.9526226734348562 | 0.9462184873949581 | 591.0   |
| truck        | 0.9478991596638655 | 0.9276315789473685 | 0.9376558603491272 | 608.0   |
| accuracy     | 0.9015             | 0.9015             | 0.9015             | 0.9015  |
| macro avg    | 0.9021288220768874 | 0.9014770263703313 | 0.9013840861536385 | 6000.0  |
| weighted avg | 0.9022969502205227 | 0.9015             | 0.9014843929551437 | 6000.0  |

#### CNN model

|              | precision          | recall             | f1-score           | support            |
|--------------|--------------------|--------------------|--------------------|--------------------|
| airplane     | 0.8645833333333334 | 0.8571428571428571 | 0.860847018150389  | 581.0              |
| automobile   | 0.9331103678929766 | 0.9425675675675675 | 0.9378151260504203 | 592.0              |
| bird         | 0.8290598290598291 | 0.7698412698412699 | 0.7983539094650206 | 630.0              |
| cat          | 0.7563176895306859 | 0.7137989778534923 | 0.7344434706397897 | 587.0              |
| deer         | 0.8064024390243902 | 0.8450479233226837 | 0.8252730109204368 | 626.0              |
| dog          | 0.7796052631578947 | 0.807495741056218  | 0.7933054393305439 | 587.0              |
| frog         | 0.8575899843505478 | 0.9102990033222591 | 0.8831587429492346 | 602.0              |
| horse        | 0.8991304347826087 | 0.8674496644295302 | 0.8830059777967549 | 596.0              |
| ship         | 0.9193825042881647 | 0.9069373942470389 | 0.9131175468483816 | 591.0              |
| truck        | 0.9073482428115016 | 0.9342105263157895 | 0.9205834683954619 | 608.0              |
| accuracy     | 0.8553333333333333 | 0.8553333333333333 | 0.8553333333333333 | 0.8553333333333333 |
| macro avg    | 0.8552530088231933 | 0.8554790925098708 | 0.8549903710546433 | 6000.0             |
| weighted avg | 0.8551000580329381 | 0.8553333333333333 | 0.8548352491855179 | 6000.0             |

Looking both tables, it is possible to classify the classes in two groups: hard-predicted classes and well-predicted classes. The first group contains the classes with low precision score (>0.8), and the other one contains classes wit better precision score (<0.8).

In the case of the Xception model (without fine-tuning), the hard-predicted set include cat, deer, dog, frog; while the other classes are part of the well-predicted group. While in the CNN model, the hard-predicted classes set include cat and dog.

Finally, in the case of the Xception model (with fine-tuning), all the precision per classes are higher than 0.8. Looking at the hard-predicted set in the model without fine-tuning, it is possible to observe that the model with fine-tuning made a great improvement in the following classes: deer, frog and cat; while the dog class is the hardest class to predict (the ROC curve shows this improvement).

![ROC curve](https://github.com/ignaciogatti/idata_challenge/blob/main/images/auc_roc_curve.png)
