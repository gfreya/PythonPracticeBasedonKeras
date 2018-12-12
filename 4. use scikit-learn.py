from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

# creat the model
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

seed = 7
np.random.seed(seed)

# load the data
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
# split input variable x and output variable Y
x = dataset[:,0:8]
Y = dataset[:,8]

# create model for scikit-learn
model = KerasClassifier(build_fn=create_model,epochs=150,batch_size=10,verbose=0)

# cross-validation
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results = cross_val_score(model,x,Y,cv=kfold)
print(results.mean())