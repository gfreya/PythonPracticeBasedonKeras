from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# creat the model
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


seed = 7
np.random.seed(seed)

# load the data
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split input variable x and output variable Y
x = dataset[:, 0:8]
Y = dataset[:, 8]

# create model for scikit-learn
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

# create parameters
param_grid = {}
param_grid['optimizer'] = ['rmsprop','adam']
param_grid['init'] = ['glorot_uniform','normal','uniform']
param_grid['epochs']=[50,100,150,200]
param_grid['batch_size'] = [5,10,20]

# parameter adjustment
grid = GridSearchCV(estimator=model,param_grid=param_grid)
results = grid.fit(x,Y)

# output the results
print('Best: %f using %s' % (results.best_score_,results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']

for mean,std,param in zip(means,stds,params):
    print('%f (%f) with: %r'%(mean,std,param))
