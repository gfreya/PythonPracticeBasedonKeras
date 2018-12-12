from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 7
np.random.seed(seed)

# load the data
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
# split input variable x and output variable Y
x = dataset[:,0:8]
Y = dataset[:,8]

kfold = StratifiedKFold(n_splits=10,random_state=seed,shuffle=True)
cvscores=[]

for train,validation in kfold.split(x,Y):
    model = Sequential()
    model.add(Dense(12,input_dim=8,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    # compilr the model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # train the model
    model.fit(x[train],Y[train],epochs=150,batch_size=10,verbose=0)

    # evaluate the model
    scores = model.evaluate(x[validation],Y[validation],verbose=0)

    # print the results
    print('%s:%.2f%%'%(model.metrics_names[1],scores[1]*100))
    cvscores.append(scores[1]*100)

# print the mean and s
print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores),np.std(cvscores)))