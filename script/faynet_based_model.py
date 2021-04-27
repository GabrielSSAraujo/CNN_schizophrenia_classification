#!/usr/bin/env python
# coding: utf-8

# ### MNIST CONVOLUTIONAL NET
# #### Criação de rede convolucional com Keras embarcado no Tensorflow para classificação de imagens de pacientes esquizofrênicos vs controle.
# #### Treinamento com  dataset Schizconnect.
# ##### Gabriel Santos e Thiago Gomes Bezerra

# |--- Imported libraries (for model creation and training) -------------------------{{{
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, save_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import PIL

print('Iniciando Modelo Baseado na FayNet....')

##----------- ATIVAÇÃO DA GPU -----------
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#----------- DEFINIÇÃO DIRETÓRIOS -----------
data_dir = 'insert your path here'

heightImage = 229
widthImage = 220

image_size = (heightImage,widthImage)
batch_size = 90
epoca = 200

#-------------- SEPARANDO DADOS PARA TREINAMENTO, VALIDAÇÂO E TESTE -----------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=500,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=500,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)
###-------- PASTAS SEPARADAS MANUALMENTE EM TREINO E TESTE ---------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     x,
#     labels='inferred',
#     label_mode='binary',
#     seed=129,
#     image_size=image_size,
#     batch_size=batch_size,
#     color_mode="grayscale",

# )
# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     x2,
#     labels='inferred',
#     label_mode='binary',
#     seed=129,
#     image_size=image_size,
#     batch_size=batch_size,
#     color_mode="grayscale",
# )
###-----------------------------------------

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
t_images = T_images.reshape(t_images.shape[0],heightImage,widthImage,1)
t_labels = np.array(Train_labels)
t_labels = t_labels.reshape(t_labels.shape[0],)

#-------------- CONJUNTO DE TESTES ----------------
Test_labels = []
Test_images = []

for images, labels in test_ds: 
    for i in range(len(images)):
      Test_images.append(images[i])
      Test_labels.append(labels[i])
te_images = np.array(Test_images)
te_images = te_images.reshape(te_images.shape[0],heightImage,widthImage,1)
te_labels = np.array(Test_labels)
te_labels = te_labels.reshape(te_labels.shape[0],)

(train_images,train_labels), (test_images,test_labels) = (t_images,t_labels),(te_images,te_labels)

train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)

test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)

inputs = np.concatenate((train_images, test_images), axis=0)
targets = np.concatenate((train_labels, test_labels), axis=0)

#-------------- IMPLEMENTANDO K-FOLD ---------------
acc_per_fold = []
loss_per_fold = []

num_folds = 5

kfold = KFold(n_splits=num_folds, shuffle=True)

#------------- AVALIAÇÃO DO MODELO DE VALIDAÇÃO CRUZADA -------------
fold_no = 1
for train, test in kfold.split(inputs, targets):
    #------------- DEFINIÇÃO DE CAMADAS DO MODELO BASEADA NA CNN FAYNET ---------------
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(229, 220, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=100, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    # model.add(layers.Dense(2, activation='softmax'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.summary()


  # metrics = [tf.keras.metrics.TruePositives(thresholds=0.5, name='TP'), tf.keras.metrics.TrueNegatives(thresholds=0.5, name='TN'), tf.keras.metrics.FalsePositives(thresholds=0.5, name='FP'), tf.keras.metrics.FalseNegatives(thresholds=0.5, name='FN'), 'accuracy']
    rms = optimizers.RMSprop(lr=0.001)#tava 2
   
    #-------------- TREINAMENTO DO MODELO -------------------------
    model.compile(loss='BinaryCrossentropy',optimizer=rms,metrics =['accuracy'])#['accuracy']
    history = model.fit(train_images, train_labels, batch_size=batch_size , epochs=epoca, verbose=0 , validation_split = 0.3 ) # bat 10 com 200 epocas; verbose=1

    plt.plot(history.history['accuracy'], label = 'Training', linewidth = 1.2)
    plt.plot(history.history['val_accuracy'], label = 'Validation', linewidth = 1.2)
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

    print(f'Training for fold {fold_no} ...')
    
    #------------- GERANDO MÉTRICAS DE DESEMPENHO-------------
    scores = model.evaluate(test_images,test_labels, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    # INCREMENTANDO NUMERO DO FOLD
    fold_no = fold_no + 1

#------------ FORNECENDO MÉDIAS ------------
print('------------------------------------------------------------------------')
print('Valor por fold:')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Media de todos os folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

#------------ SALVANDO MODELO ---------------
print('Salvando o modelo...')
model.save('fay.h5')
print('Modelo salvo!')