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

* <b>Sample Images</b> --> up (Negative) | down (Positive)

    ![1](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/negative.png)

    ![2](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/positive.png)

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
