import os
import pathlib
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
from keras.preprocessing.image import *
from tensorflow_hub import KerasLayer