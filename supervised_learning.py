import os
import cv2
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

# Get all images

emoji_types = ['Happy', 'Sad', 'Angry', 'Poo', 'Surprised']
X = []
y = []
base_path = os.path.join(os.getcwd(), "images_nor")
image_folders = os.listdir(base_path)
for folder in image_folders:
    folder_path = os.path.join(base_path, folder)
    files = os.listdir(folder_path)
    for file in files:
        if file == '.DS_Store': continue
        img = Image.open(os.path.join(folder_path, file))
        img = img.convert('L')
        img_array = np.array(img)
        x_shape, y_shape = img_array.shape
        img_array = img_array.reshape(x_shape * y_shape)
        X.append(img_array)
        y.append(emoji_types.index(folder))

X = np.asarray(X)
y = np.asarray(y)

# KNN
knn = KNeighborsClassifier(3)
knn_scores = cross_val_score(knn, X, y, cv = 5)
print("KNN Accuracy: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))

# RB-SVM

rb_svm = SVC(gamma=2, C=1)
rb_svm_scores = cross_val_score(rb_svm, X, y, cv = 5)
print("RB-SVM Accuracy: %0.2f (+/- %0.2f)" % (rb_svm_scores.mean(), rb_svm_scores.std() * 2))

# Convolutional Neural Network
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)
X_train_copy = X_train.copy().reshape(2400, 32, 32, 1)
X_test_copy = X_test.copy().reshape(601, 32, 32, 1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)
input_shape = (32, 32, 1)
batch_size = 50
epochs = 12

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train_copy, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_copy, y_test))
score = model.evaluate(X_test_copy, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(model.predict(X_test_copy, verbose=1))
