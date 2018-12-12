from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

# set random seed
seed = 7
np.random.seed(seed)

# load the data
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

x_train,x_increment,Y_train,Y_increment = train_test_split(x,Y,test_size=0.2,random_state=seed)

# transfer to category
Y_train_labels = to_categorical(Y,num_classes=3)

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
model.fit(x_train,Y_train_labels,epochs=10,batch_size=5,verbose=0)

scores = model.evaluate(x,Y_train_labels,verbose=0)
print('Base %s: %.2f%%' % (model.metrics_names[1],scores[1]*100))

# save the model to json file
model_json = model.to_json()
with open('model.increment.json','w') as file:
    file.write(model_json)

# save the weights in the model
model.save_weights('model.increment.json.h5')

# load the model from json file
with open('model.invrement.json','r') as file:
    model_json = file.read()

# load the model
new_model = model_from_json(model_json)
new_model.load_weights('model.increment.json.h5')

# incremental model
# compile the model
Y_increment_labels = to_categorical(Y_increment,num_classes=3)
new_model.fit(x_increment,Y_increment_labels,epochs=10,batch_size=5,verbode=2)

# evaluate the model
scores = new_model.evaluate(x_increment,Y_increment_labels,verbose=0)
print('Increment %s:%.2f' % (model.metrics_names[1],scores[1]*100))
