# TransferLearning
Feature extraction and Fine-tuning neural networks

# Overview
* The steps of this project are the following:
    * Extract features from images using pre-traineed network
    * Using extracted features to train a classifier
    * Fine-tuning a network using the Keras API
    * Fine-tuning a network using tensorflow

#
## Running feature_extractor.py
* <b>Extract features</b> from images using pre-traineed network.
* Dataset LINK : <b>https://jovian.com/outlink?url=https%3A%2F%2Fs3.amazonaws.com%2Ffast-ai-imageclas%2Fstanford-cars.tgz</b>

* <b>Sample Images</b>

    ![1](https://github.com/hasanoqool/TransferLearning/blob/main/images/00001.jpg)
    ![2](https://github.com/hasanoqool/TransferLearning/blob/main/images/00002.jpg)
    ![3](https://github.com/hasanoqool/TransferLearning/blob/main/images/00003.jpg)

#
## Running feature_extractor.py
* <b>Extract features</b> from images using pre-traineed network.

#
## Running fine_tune_keras.py
* <b>Fine-tuning</b> a network using the Keras API.
* Dataset LINK : <b>http://www.robots.ox.ac.uk/~vgg/data/flowers/17</b>

* <b>Sample Images</b>

    ![1](https://github.com/hasanoqool/TransferLearning/blob/main/images/1.jpg)
    ![2](https://github.com/hasanoqool/TransferLearning/blob/main/images/2.jpg)
    ![3](https://github.com/hasanoqool/TransferLearning/blob/main/images/3.jpg)

* <b>Without fine-tune</b>
    | Test Loss  |  Test accuracy |
    | ------------- | ------------- |
    |  0.1398 |  0.8602 |

* <b>With fine-tune</b>
    | Test Loss  |  Test accuracy |
    | ------------- | ------------- |
    |  0.0773 |  0.9227 |
#
## Running fine_tune_tf.py
* <b>Fine-tuning</b> a network using tensorflow.
* Dataset LINK : <b>http://www.robots.ox.ac.uk/~vgg/data/flowers/17</b>

* <b>Sample Images</b>

    ![1](https://github.com/hasanoqool/TransferLearning/blob/main/images/4.jpg)
    ![2](https://github.com/hasanoqool/TransferLearning/blob/main/images/5.jpg)
    ![3](https://github.com/hasanoqool/TransferLearning/blob/main/images/6.jpg)

    | Test Loss  |  Test accuracy |
    | ------------- | ------------- |
    |  0.0625 |  0.9375 |
#
## Contact
* Reach me out here: https://www.linkedin.com/in/hasanoqool/
#