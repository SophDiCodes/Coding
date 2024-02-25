#Project Made With Guidance of InspiritAI Program
#@title Run this to download data and prepare our environment! { display-mode: "form" }
from google.colab.output import eval_js

import time
start_time = time.time()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.notebook import tqdm

import keras
from keras import backend as K
from tensorflow.keras.layers import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import random
from PIL import Image
import gdown

import argparse
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import struct
from google.colab.patches import cv2_imshow
from copy import deepcopy
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from keras.applications.mobilenet import MobileNet

!pip install hypopt
from hypopt import GridSearch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

!pip install -U opencv-contrib-python
import cv2

!pip install tensorflowjs
import tensorflowjs as tfjs

from google.colab import files

import requests, io, zipfile

# Prepare data
images_1 = os.makedirs('images_1', exist_ok=True)
images_2= os.makedirs('images_2', exist_ok=True)
images_all= os.makedirs('images_all', exist_ok=True)

metadata_path = 'metadata.csv'
image_path_1 = 'images_1.zip'
image_path_2 = 'images_2.zip'
images_rgb_path = 'hmnist_8_8_RGB.csv'

!wget -O metadata.csv 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/metadata.csv'
!wget -O images_1.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_1.zip'
!wget -O images_2.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_2.zip'
!wget -O hmnist_8_8_RGB.csv 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/hmnist_8_8_RGB.csv'
!unzip -q -o images_1.zip -d images_1
!unzip -q -o images_2.zip -d images_2

!pip install patool
import patoolib

import os.path
from os import path

from distutils.dir_util import copy_tree

fromDirectory = 'images_1'
toDirectory = 'images_all'

copy_tree(fromDirectory, toDirectory)

fromDirectory = 'images_2'
toDirectory = 'images_all'

copy_tree(fromDirectory, toDirectory)

print("Downloaded Data")

#Preparing Dataset for Analysis
IMG_WIDTH = 100
IMG_HEIGHT = 75

X = []
X_gray = []

y = []

#Initializes X, X_gray, and y variables
metadata = pd.read_csv(metadata_path)
metadata['category'] = metadata['dx'].replace({'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6,})


for i in tqdm(range(len(metadata))):
  image_meta = metadata.iloc[i]
  path = os.path.join(toDirectory, image_meta['image_id'] + '.jpg')
  img = cv2.imread(path,cv2.IMREAD_COLOR)
  img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))

  img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  X_gray.append(img_g)

  X.append(img)
  y.append(image_meta['category'])

X_gray = np.array(X_gray)
X = np.array(X)
y = np.array(y)

#Shows shape of updated X, X_gray, y variables
X.shape, X_gray.shape, y.shape

#Plots Distribution of Dataset
objects = ('akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc')
y_pos = np.arange(len(objects))
occurances = []

for obj in objects:
  occurances.append(np.count_nonzero(obj == metadata['dx']))

print(occurances)

plt.bar(y_pos, occurances, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Samples')
plt.title('Distribution of Classes Within Dataset')

plt.show()

#Reducing Dataset Size
sample_cap = 1113
option = 2

#Also Reduces Dataset Size, only reduces number
#of nv samples to be the same amount as the number of samples
#found int he second most prevalent class
if (option == 2):
  objects = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
  class_totals = [0,0,0,0,0,0,0]

  for i in range(len(X)):
    class_totals[y[i]] += 1

  print("Initial Class Samples")
  print(class_totals)

  largest_index = class_totals.index(max(class_totals))
  class_totals[largest_index] = 0

  second_largest_val = max(class_totals)

  indicies = []
  iter = 0
  for i in range(len(X)):
    if y[i] == largest_index:
      if iter != second_largest_val:
        indicies.append(i)
        iter += 1
      else:
        continue
    else:
      indicies.append(i)

  class_totals = [0,0,0,0,0,0,0]

  for i in range(len(X)):
    class_totals[y[i]] += 1

  print("Modified Class Samples")
  print(class_totals)

  X = X[indicies]
  X_gray = X_gray[indicies]

  y = y[indicies]

else:
  print("This option was not selected")

#Data Augmentation

#Test/train split for grayscale image data and color image data
X_train, X_test, X_gray_train, X_gray_test, y_train, y_test = train_test_split(X, X_gray, y, test_size=0.4, random_state=101)

#Iterates through all images to create duplicate with random transformation, doubling training dataset size
X_augmented = []
X_gray_augmented = []

y_augmented = []

for i in tqdm(range(len(X_train))):
  transform = random.randint(0,1)
  if (transform == 0):
    # Flip the image across the y-axis

    #YOUR CODE HERE
    flipped_image = cv2.flip(X_train[i], -1)
    flipped_image_gray = cv2.flip(X_gray_train[i], -1)

    #End code

    X_augmented.append(flipped_image)
    X_gray_augmented.append(flipped_image_gray)
    y_augmented.append(y_train[i])
  else:
    # Zoom 33% into the image
    #YOUR CODE HERE
    blurred_image = cv2.blur(X_train[i], (3,3))
    blurred_image_gray = cv2.blur(X_gray_train[i], (3,3))

    #End code

    X_augmented.append(blurred_image)
    X_gray_augmented.append(blurred_image_gray)
    y_augmented.append(y_train[i])

#Combines Augmented Data with Existing Samples
X_augmented = np.array(X_augmented)
X_gray_augmented = np.array(X_gray_augmented)

y_augmented = np.array(y_augmented)

X_train = np.vstack((X_train,X_augmented))
X_gray_train = np.vstack((X_gray_train,X_gray_augmented))

y_train = np.append(y_train,y_augmented)

#Creating Basic Machine Learnign Mmodels
knn = KNeighborsClassifier(n_neighbors=5)

#Image Flattening with grayscale image data
X_g_train_flat = X_gray_train.reshape(X_gray_train.shape[0],-1)
X_g_test_flat = X_gray_test.reshape(X_gray_test.shape[0],-1)
print (X_g_train_flat.shape)
print (X_g_test_flat.shape)

#train models on flattened grayscale images
knn.fit(X_g_train_flat, y_train)

def model_stats(name, y_test, y_pred, y_pred_proba):
  cm = confusion_matrix(y_test, y_pred)

  print(name)

  accuracy = accuracy_score(y_test, y_pred)
  print ("The accuracy of the model is " + str(round(accuracy,5)))

  multi_class = "ovo"
  roc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')

  print ("The ROC AUC Score of the model is " + str(round(roc_score,5)))

  return cm

print(y_test)

y_pred = knn.predict(X_g_test_flat)
y_pred_proba = knn.predict_proba(X_g_test_flat)
model_stats('KNN', y_test, y_pred, y_pred_proba)

#confusion matrices
def plot_cm(name, cm):
  classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

  df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
  df_cm = df_cm.round(5)

  plt.figure(figsize = (12,8))
  sns.heatmap(df_cm, annot=True, fmt='g')
  plt.title(name)
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.show()

stats = model_stats('KNN', y_test, y_pred, y_pred_proba)
plot_cm('KNN', stats)
