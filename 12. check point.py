from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# load data
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target
Y_labels = to_categorical(Y,num_classes=3)

seed = 7
np.random.seed(seed)

# create model
def create_model(optimizer='adam',init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    # compile the model
    model.compile(loss='catogorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model

model = create_model()

# set the check point
filepath = 'weights-improvement-{epoch:02d}-{val-acc:.2f}.h5'
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val-acc',verbose=1,save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(x,Y_labels,validation_split=0.2,epochs=200,batch_size=5,verbose=0,callbacks=callback_list)

