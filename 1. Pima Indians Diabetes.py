from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# set random seed
np.random.seed(7)

# load the data
# use loadtst() to load Pima Indians data
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
# split x and y
x = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# train the model
model.fit(x=x,y=Y,epochs=150,batch_size=10,validation_split=0.2)

# evaluate the model
scores = model.evaluate(x=x,y=Y)
print('\n%s:%.2f%%' %(model.metrics_names[1],scores[1]*100))


"""
1. load the data 
2. define model
3. compile model
4. train model
5. evaluate model
When evaluating the model, it is better to split the dataset into training data and testing data
"""