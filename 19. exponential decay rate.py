from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from math import pow,floor

# load the data
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

# set random seed
seed = 7
np.random.seed(seed)

# calculate the learning rate
def step_decay(epoch):
    init_lrate = 0.1
    drop = 0.5
    epochs_drop = 10
    lrate = init_lrate*pow(drop,floor(1+epoch)/epochs_drop)
    return lrate

# create model
def create_model(init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    # optimization of the model
    learningRate = 0.1
    momentum = 0.9
    decay_rate = 0.0
    sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)

    # compile the model
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    return model

# learning rate exponential recall
lrate = LearningRateScheduler(step_decay)
epochs = 200
model = KerasClassifier(build_fn=create_model,epochs=epochs,batch_size=5,verbose=1,callbacks=[lrate])
model.fit(x,Y)