code breakthrought:


Below is the structured code organized as a project directory:

project_simpsons_recognition/
│
├── data/
│   ├── train/
│   │   ├── character_1/
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   ├── character_2/
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       └── test_image.jpg
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── model.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore




1) preprocess.py:


import os
import caer

def preprocess_data(char_path, IMG_SIZE=(80, 80), channels=1):
    char_dict = {}
    for char in os.listdir(char_path):
        char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

    char_dict = caer.sort_dict(char_dict, descending=True)
    characters = [i[10] for i in char_dict[:10]]

    train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)
    featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)
    featureSet = caer.normalize(featureSet)
    labels = caer.to_categorical(labels, len(characters))

    return featureSet, labels, characters

2) model.py:

import canaro
from keras.utils import to_categorical

def create_and_train_model(x_train, y_train, x_val, y_val, characters, IMG_SIZE=(80, 80), channels=1):
    BATCH_SIZE = 32
    EPOCH = 10

    datagen = canaro.generators.imageDataGenerator()
    train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

    model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE,
                                               channels=channels,
                                               output_dim=len(characters),
                                               loss='binary_crossentropy',
                                               decay=1e-6,
                                               learning_rate=0.001,
                                               momentum=0.9,
                                               nesterov=True)

    training = model.fit(train_gen,
                         steps_per_epoch=len(x_train) // BATCH_SIZE,
                         epochs=EPOCH,
                         validation_data=(x_val, y_val),
                         validation_steps=len(y_val) // BATCH_SIZE)

    return model

3) main.py:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from model import create_and_train_model

def prepare_image(img, IMG_SIZE=(80, 80)):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = np.reshape(img, (*IMG_SIZE, 1))
    return img

def main():
    char_path = 'data/train/'
    test_path = 'data/test/test_image.jpg'

    x_train, y_train, characters = preprocess_data(char_path)

    x_val, y_val = x_train[-500:], y_train[-500:]
    x_train, y_train = x_train[:-500], y_train[:-500]

    model = create_and_train_model(x_train, y_train, x_val, y_val, characters)

    img = cv.imread(test_path)
    plt.imshow(img, cmap='gray')
    plt.show()

    prepared_img = prepare_image(img)
    predictions = model.predict(np.array([prepared_img]))
    print(characters[np.argmax(predictions[0])])

if __name__ == "__main__":
    main()
