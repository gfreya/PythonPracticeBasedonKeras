from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json

# load the data
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

# transfer to category
Y_labels = to_categorical(Y,num_classes=3)

# set random seed
seed = 7
np.random.seed(seed)

# create the model
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

# save the model to json file
model_json = model.to_json()
with open('model.json','w') as file:
    file.write(model_json)

# save the weights in the model
model.save_weights('model.json.h5')

# load the model from json file
with open('model.json','r') as file:
    model_json = file.read()

# load the model
new_model = model_from_json(model_json)
new_model.load_weights('model.json.h5')

# compile the model
new_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# evaluate the model
scores = new_model.evaluate(x,Y_labels,verbose=0)
print('%s:%.2f' % (model.metrics_names[1],scores[1]*100))
