# TransferLearning
Feature extraction and Fine-tuning neural networks

# Overview
* The steps of this project are the following:
    * Extract features from images using pre-traineed network
    * Using extracted features to train a classifier
    * Train a classifier to classify multi watches (Multi-Label)
    * Implementing ResNet from scratch
    * Classify images using a pre-trained network using the Keras 

#
## Running feature_extractor.py
* <b>Extract features/b> from images using pre-traineed network.
* Dataset LINK : <b>https://jovian.com/outlink?url=https%3A%2F%2Fs3.amazonaws.com%2Ffast-ai-imageclas%2Fstanford-cars.tgz</b>

* <b>Sample Images</b>

    ![1](https://github.com/hasanoqool/TransferLearning/blob/main/images/00001.jpg)
    ![2](https://github.com/hasanoqool/TransferLearning/blob/main/images/00002.jpg)
    ![3](https://github.com/hasanoqool/TransferLearning/blob/main/images/00003.jpg)

* <b>Model Evaluation</b>:

    | train_loss  |  train_accuracy |
    | ------------- | ------------- |
    |  0.5435 |  0.8920 |

    | val_loss  |  val_accuracy |
    | ------------- | ------------- |
    |  0.2501 |  0.9193 |

    | test_loss  |  test_accuracy |
    | ------------- | ------------- |
    |  0.2077 |  0.9225 |
#
