from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import backend
from scipy.misc import toimage
import numpy as np
backend.set_image_data_format('channels_first')

# load the data
(X_train,y_train),(X_validation,y_validation)=cifar10.load_data()

for i in range(0,9):
    plt.subplot(331+i)
    plt.imshow(toimage(X_train[i]))

plt.show()

# set random seed
seed = 7
np.random.seed(seed)

# normalize data 0-1
X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_train = X_train/255.0
X_validation = X_validation/255.0

# one-hot code
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]

def create_model(epochs):
    model = Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(3,32,32),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

epochs = 25
model = create_model(epochs)
model.fit(x=X_train,y=y_train,epochs=epochs,batch_size=32,verbose=2)
scores = model.evaluate(x=X_validation,y=y_validation,verbose=0)
print('Accuracy:%.2f%%'%(scores[1]*100))