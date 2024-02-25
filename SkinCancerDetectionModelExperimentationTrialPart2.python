#started at inspirit AI, still finishing
#prepares environment
print("Installing packages...")
!pip -q install hypopt tensorflowjs > /dev/null
!pip -q install git+https://github.com/rdk2132/scikeras # workaround for scikeras deprecation
import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras.api.keras as keras
import scikeras
import tensorflowjs as tfjs

from tqdm.notebook import tqdm
from keras.layers import * # import all, including Dense, add, Flatten, etc.
from keras.models import Model, Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from keras.applications.mobilenet import MobileNet
from hypopt import GridSearch

print("Downloading files...")
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/X.npy'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/X_g.npy'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/y.npy'

# Set up Web App
os.makedirs("static/js", exist_ok=True)
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/skin_cancer_diagnosis_script.js' &> /dev/null
output = 'static/js/skin_cancer_diagnosis_script.js'

print("Done!")

#loads last part's data
X = np.load("X.npy")
X_g = np.load("X_g.npy")
y = np.load("y.npy")

#Performs Data Augmentation
IMG_WIDTH = 100
IMG_HEIGHT = 75

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
X_g_train, X_g_test, y_train, y_test = train_test_split(X_g, y, test_size=0.4, random_state=101)

X_augmented = []
X_g_augmented = []

y_augmented = []

for i in tqdm(range(len(X_train))):
  transform = random.randint(0, 1)
  if (transform == 0):
    # Flip the image across the y-axis
    X_augmented.append(cv2.flip(X_train[i], 1))
    X_g_augmented.append(cv2.flip(X_g_train[i], 1))
    y_augmented.append(y_train[i])
  else:
    zoom = 0.33 # Zoom 33% into the image

    centerX, centerY = int(IMG_HEIGHT/2), int(IMG_WIDTH/2)
    radiusX, radiusY = int((1-zoom)*IMG_HEIGHT*2), int((1-zoom)*IMG_WIDTH*2)

    minX, maxX = centerX-radiusX, centerX+radiusX
    minY, maxY = centerY-radiusY, centerY+radiusY

    cropped = (X_train[i])[minX:maxX,  minY:maxY]
    new_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
    X_augmented.append(new_img)

    cropped = (X_g_train[i])[minX:maxX, minY:maxY]
    new_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
    X_g_augmented.append(new_img)

    y_augmented.append(y_train[i])

X_augmented = np.array(X_augmented)
X_g_augmented = np.array(X_g_augmented)

y_augmented = np.array(y_augmented)

X_train = np.vstack((X_train, X_augmented))
X_g_train = np.vstack((X_g_train, X_g_augmented))

y_train = np.append(y_train, y_augmented)

#CNN Model
def CNNClassifier(epochs=20, batch_size=10, layers=5, dropout=0.5, activation='relu'):
  def set_params():
    i = 1  
  def create_model():
    model = Sequential()
    
    for i in range(layers):
      model.add(Conv2D(64, (3, 3), padding='same'))
      model.add(Activation(activation))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout / 2.0))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation(activation))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout / 2.0))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    # Let's train the model using RMSprop
    return model
  opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
  return KerasClassifier(model=create_model, optimizer=opt, loss='categorical_crossentropy', epochs=epochs, batch_size=batch_size, verbose=1, validation_batch_size=batch_size, validation_split=.4, metrics=['accuracy'])

#processes x variables, transforms y labels into one hot encoded labels for training
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

y_train_onehot = np.zeros((y_train.size, y_train.max().astype(int)+1))
y_train_onehot[np.arange(y_train.size), y_train.astype(int)] = 1
y_train_onehot = y_train_onehot.astype(np.float32)

y_test_onehot = np.zeros((y_test.size, y_test.max().astype(int)+1))
y_test_onehot[np.arange(y_test.size), y_test.astype(int)] = 1
y_test_onehot = y_test_onehot.astype(np.float32)

cnn = CNNClassifier()
cnn.fit(None, None,
        validation_data=(None, None))

tfjs.converters.save_keras_model(cnn.model_, 'cnn_model')

#defines model_stats()
def model_stats(name, y_test, y_pred, y_pred_proba):
  y_pred_1d = [0] * len(y_test)
  for i in range(len(y_test)):
    y_pred_1d[i] = np.where(y_pred[i] == 1)[0][0]

  cm = confusion_matrix(y_test, y_pred_1d)

  print(name)

  accuracy = accuracy_score(y_test, y_pred_1d)
  print ("The accuracy of the model is " + str(round(accuracy, 5)))

  y_test_onehot = np.zeros((y_test.size, y_test.max().astype(int)+1))
  y_test_onehot[np.arange(y_test.size), y_test.astype(int)] = 1

  roc_score = roc_auc_score(y_test_onehot, y_pred_proba)

  print ("The ROC AUC Score of the model is " + str(round(roc_score, 5)))
  
  return cm

#redefining plot_cm() from part 1
def plot_cm(name, cm):
  classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

  df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
  df_cm = df_cm.round(5)

  plt.figure(figsize = (12, 8))
  sns.heatmap(df_cm,  annot=True, fmt='g')
  plt.title(name + " Model Confusion Matrix")
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.show()

#unfinished! still working on it!
