import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense

seed = 7
batch_size = 2
epochs = 400
filename = 'international-airline-passages.csv'
footer = 3
look_back = 3

def create_dataset(dataset):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back-1):
        x = dataset[i:i+look_back,0]
        dataX.append(x)
        y = dataset[i+look_back,0]
        dataY.append(y)
        print('X: %s, Y:%s'%(x,y))
    return np.array(dataX),np.array(dataY)

def build_model():
    model = Sequential()
    model.add(Dense(units=12,input_dim=look_back,activation='relu'))
    model.add(Dense(units=8,activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

if __name__ == '__main__':

    #set random seed
    np.random.seed(seed)

    # load the data
    data = read_csv(filename,usecols=[1],engine='python',skipfooter=footer)
    dataset = data.values.astype('float32')
    train_size = int(len(dataset)*0.67)
    validation_size = len(dataset)-train_size
    train,validation = dataset[0:train_size,:],dataset[train_size:len(dataset),:]

    # create dataset, make data correlate to each other
    X_train,y_train = create_dataset(train)
    X_validation,y_validation = create_dataset(validation)

    # train the model
    model = build_model()
    model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=2)

    # evaluate the model
    train_score = model.evaluate(X_train,y_train,verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' %(train_score,math.sqrt(train_score)))
    validation_score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Validation Score: %.2f MSE (%.2f RMSE)' % (validation_score, math.sqrt(validation_score)))


    #prediction clination
    predict_train = model.predict(X_train)
    predict_validation = model.predict(X_validation)

    # training data plot
    predict_train_plot = np.empty_like(dataset)
    predict_train_plot[:,:] = np.nan
    predict_train_plot[look_back:len(predict_train)+look_back,:] = predict_train

    # validation data plot
    predict_validation_plot = np.empty_like(dataset)
    predict_validation_plot[:,:]=np.nan
    predict_validation_plot[len(predict_train)+look_back*2+1:len(dataset)-1,:] = predict_validation

    # show the plot
    plt.plot(dataset,color='blue')
    plt.plot(predict_train_plot,color='green')
    plt.plot(predict_validation_plot,color='red')
    plt.show()