from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

seed = 7
np.random.seed(seed)

# load the data
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
# split x and y
x = dataset[:,0:8]
Y = dataset[:,8]

# split the dataset
x_train,x_validation,Y_train,Y_validation = train_test_split(x,Y,test_size=0.2,random_state=seed)

# create the model
model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# train the model
model.fit(x_train,Y_train,validation_data=(x_validation,Y_validation),epochs=150,batch_size=10)