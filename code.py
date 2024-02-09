import os
import cv2 as cv
import numpy as np
import gc
import canaro
import caer
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

IMG_SIZE=(80,80)
channels=1
char_path=r'/Users/name/location/'
char_dict={}
for char in os.listdir(char_path):
    char_dict[char]=len(os.listdir(os.path.join(char_path,char)))

# sorting in descending order
char_dict=caer.sort_dict(char_dict,descending=True)
characters=[]
count=0
for i in char_dict:
    characters.append(i[10])
    count+=1
    if count>=10:
        break

# creating the training data
train=caer.preprocess_from_dir(char_path,characters,channels=channels,IMG_SIZE=IMG_SIZE,isShuffle=True)

featureSet,labels=caer.sep_train(train,IMG_SIZE=IMG_SIZE)

# normalize the feature set
featureSet=caer.normalize(featureSet)
labels=to_categorical(labels,len(characters))
x_train,x_val,y_train,y_val=caer.train_val_split(featureSet,labels,val_ratio=0.2)

del train
del featureSet
del labels
gc.collect()

# image data generator
BATCH_SIZE=32
EPOCH=10
datagen=canaro.generators.imageDataGenerator()
train_gen=datagen.flow(x_train,y_train,batch_size=BATCH_SIZE)

# creating a model
model=canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE,
                                        channels=channels,
                                        output_dim=len(characters),
                                        loss='binary_crossentropy',
                                        decay=1e-6,
                                        learning_rate=0.001,
                                        momentum=0.9,
                                        nesterov=True)
model.summary()

callback_list=[LearningRateScheduler(canaro.lr_schedule)]

# training the model
training=model.fit(train_gen,
                   steps_per_epoch=len(x_train)//BATCH_SIZE,
                   epochs=EPOCH,
                   validation_data=(x_val,y_val),
                   validation_steps=len(y_val)//BATCH_SIZE,
                   callbacks=callback_list)

# Load and process the test image
test_path=r'path_to_your_test_image.jpg'
img=cv.imread(test_path)
plt.imshow(img,cmap='gray')
plt.show()

def prepare(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img=cv.resize(img,IMG_SIZE)
    img=np.reshape(img,(*IMG_SIZE,1)) # Fixing reshape function
    return img

# Make predictions
predictions=model.predict(np.array([prepare(img)]))
print(characters[np.argmax(predictions[0])])
