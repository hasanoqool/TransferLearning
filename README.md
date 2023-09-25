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
## Running detect_smiles.py
* Train a binary classifier to classify face reactions (positive, negative) on the <b>SMILEs dataset</b>.
* Dataset LINK : <b>https://github.com/hromi/SMILEsmileD/tree/master</b>

* <b>Sample Images</b> --> up (Negative) | down (Positive)

    ![Negative](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/negative.png)

    ![Positive](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/positive.png)

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
