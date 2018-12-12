from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# load data
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target
Y_labels = to_categorical(Y,num_classes=3)

seed = 7
np.random.seed(seed)

# create model
def load_model(optimizer='adam',init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    filepath = 'weights.best.h5'
    model.load_weights(filepath=filepath)

    # compile the model
    model.compile(loss='catogorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model

model = load_model()

scores = model.evaluate(x, Y_labels,verbose=0)
print('%s:%.2f%%' % (model.metrics_names[1],scores[1]*100))
