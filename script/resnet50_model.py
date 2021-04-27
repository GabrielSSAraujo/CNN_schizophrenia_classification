#!/usr/bin/env python
# coding: utf-8

# ### MNIST CONVOLUTIONAL NET
# #### Criação de rede convolucional com Keras embarcado no Tensorflow para classificação de imagens de pacientes esquizofrênicos vs controle.
# #### Treinamento com  dataset Schizconnect.
# ##### Gabriel Santos e Thiago Gomes Bezerra

# |--- Imported libraries (for model creation and training) ---|----------------------{{{
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.compat.v1 import ConfigProto
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

##----------- ATIVAÇÃO DA GPU -----------
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#----------- DEFINIÇÃO DIRETÓRIOS -----------

data_dir_training = 'insert your training path here'
data_dir_testing = 'insert your test path here'

heightImage = 229
widthImage = 220
batch_size = 40
epoca = 200

image_size = (heightImage,widthImage)

#-------------- SEPARANDO DADOS PARA TREINAMENTO, VALIDAÇÂO E TESTE -----------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_training,
    labels='inferred',
    label_mode='binary',
    seed=500,
    batch_size = batch_size,
    image_size = image_size,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_testing,
    labels='inferred',
    label_mode='binary',
    seed=120,
    batch_size = batch_size,
    image_size = image_size,
)

#-------------- VISUALIZANDO CLASSES ENCONTRADAS NOS DIRETÓRIOS ---------------
class_names = train_ds.class_names
print(class_names)

#-------------- LENDO, SALVANDO DADOS DE ENTRADA -----------------
#-------------- CONJUNTO DE TREINAMENTO E VALIDAÇÃO ------------
Train_labels = []
Train_images = []

for images, labels in train_ds:
    for i in range(len(images)):
      Train_images.append(images[i])
      Train_labels.append(labels[i])
t_images = np.array(Train_images)
t_images = t_images.reshape(t_images.shape[0],229,220,3)
t_labels = np.array(Train_labels)

t_labels = t_labels.reshape(t_labels.shape[0],)

#-------------- LENDO, SALVANDO DADOS DE ENTRADA -----------------
#-------------- CONJUNTO DE TREINAMENTO ------------
Test_labels = []
Test_images = []
for images, labels in test_ds:
    for i in range(len(images)):
      Test_images.append(images[i])
      Test_labels.append(labels[i])
te_images = np.array(Test_images)
te_images = te_images.reshape(te_images.shape[0],heightImage,widthImage,3)
te_labels = np.array(Test_labels)
te_labels = te_labels.reshape(te_labels.shape[0],)

(train_images,train_labels), (test_images,test_labels) = (t_images,t_labels),(te_images,te_labels) 
train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)

test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)

#-------------- DEFINIÇÃO DO MODELO RESNET50 -------------------------
restnet = ResNet50(include_top=False, weights="imagenet", input_shape = (heightImage, widthImage, 3))
for layer in restnet.layers:
    layer.trainable = False

restnet.summary()
input_shape = image_size
model = models.Sequential()
model.add(restnet)

#-------------- DEFINIÇÃO DAS CAMADAS DE SAIDA -------------------------
model.add(layers.Dense(200, activation='relu', input_dim=input_shape))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))
model.summary()

#-------------- TREINAMENTO DO MODELO -------------------------
rms = optimizers.RMSprop(lr=0.0001)
model.compile(loss='binary_crossentropy',optimizer=rms,metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")])
history = model.fit(train_images,train_labels, batch_size=batch_size, epochs=epoca, verbose=1 , validation_split=0.3)
plt.plot(history.history['binary_accuracy'], label = 'Training', linewidth = 1.2)
plt.plot(history.history['val_binary_accuracy'], label = 'Validation', linewidth = 1.2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.show()
plt.plot(history.history['loss'], label = 'Training', linewidth = 1.2)
plt.plot(history.history['val_loss'], label = 'Validation', linewidth = 1.2)
plt.xlabel('Epoch')
plt.ylabel('Loss function')
plt.legend(loc="upper left")
plt.show()

#------------- MÉTRICAS DE DESEMPENHO-------------
test_loss,test_acc = model.evaluate(test_images,test_labels)
print("Test Accuracy Validacao: ", test_acc)
print("Test Loss Validacao: ", test_loss)