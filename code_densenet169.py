import os
import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l1_l2

# Définir le chemin vers le répertoire contenant vos données
data_directory = '..\Datasets\TrashType_Image_Dataset'

# Vérifier si le répertoire de données existe
if not os.path.exists(data_directory):
    print("Le répertoire de données n'existe pas. Veuillez spécifier le chemin correct.")
    exit()

# Charger les données
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisation
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Spécifier le fractionnement validation/train ici
)

train_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # 'sparse' car nous n'utilisons pas de one-hot encoding
    subset='training'  # Définir comme ensemble d'entraînement
)

validation_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation'  # Définir comme ensemble de validation
)

# Construire le modèle
base_model = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Dégeler les dernières couches du modèle de base
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Ajouter des couches personnalisées avec régularisation
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

# Afficher les courbes d'accuracy et de perte
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()