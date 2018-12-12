from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_yaml

# load the data
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

# labels--> catogory
Y_labels = to_categorical(Y,num_classes=3)

# set random seed
seed = 7
np.random.seed(seed)

# create model
def create_model(optimizer='rmsprop',init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    # compile the model
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

model = create_model()
model.fit(x,Y_labels,epochs=200,batch_size=5,verbose=0)
scores = model.evaluate(x,Y_labels,verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1],scores[1]*100))

# save to yaml
model_yaml = model.to_yaml()
with open('model.yaml','w') as file:
    file.write(model_yaml)

# save the weights of model
model.save_weights('model.yaml.h5')

# load the model from json
with open('model.yaml','r') as file:
    model_yaml = file.read()

# load the model
new_model = model_from_yaml(model_yaml)
new_model.load_weights('model.yaml.h5')

# compile the model
new_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# evaluate the model
scores = new_model.evaluate(x,Y_labels,verbose=0)
print('%s:%.2f' % (model.metrics_names[1],scores[1]*100))
