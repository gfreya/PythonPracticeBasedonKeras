from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
# load the data
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

# set random seed
seed = 7
np.random.seed(seed)

# create model
def create_model(init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    # optimize the model
    learningRate = 0.1
    momentum = 0.9
    decay_rate = 0.005

    # define decay rate
    sgd = SGD(lr=learningRate,momentum=momentum,decay=decay_rate,nesterov=False)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

epochs = 200
model = KerasClassifier(build_fn=create_model,epochs=epochs,batch_size=5,verbose=1)
model.fit(x,Y)