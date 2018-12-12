from pandas import read_csv,DataFrame,concat
from datetime import datetime
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

batch_size = 72
epochs = 50
# predict by previous data
n_input = 1
n_train_hours = 365*24*4
n_validation_hours = 24*5
filename = 'pollution_original_csv'

def parse(x):
    return datetime.strftime(x,'%Y %m %d %H')

def load_dataset():
    # load the data
    dataset = read_csv(filename,parse_dates=[['year','month','day','hour']],index_col=0,date_parser=parse)

    # delete No. comlumn
    dataset.drop('No',axis=1,inplace=True)

    # set the column name
    dataset.columns = ['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
    dataset.index.name='date'

    # use mean num to padding
    dataset['pollution'].fillna(dataset['pollution'].mean(),inplace=True)

    return dataset

def convert_dataset(data,n_input,out_index=0,dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols,names = [],[]
    # input sequence (t-n,...,t-1)
    for i in range(n_input,0,-1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d'%(j+1,i)) for j in range(n_vars)]
    # output
    cols.append(df[df.columns[out_index]])
    names += ['results']
    # concat input/output column
    result = concat(cols, axis=1)
    result.columns = names
    # delete nan
    if dropnan:
        result.dropna(inplace=True)
    return result

# class_indexs encode
def class_encode(data,class_indexs):
    encoder = LabelEncoder()
    class_indexs = class_indexs if type(class_indexs) is list else [class_indexs]
    values = DataFrame(data).values
    for index in class_indexs:
        values[:,index] = encoder.fit_transform(values[:,index])
    return DataFrame(values) if type(data) is DataFrame else values

def build_model(lstm_input_shape):
    model = Sequential()
    model.add(LSTM(units=50,input_shape=lstm_input_shape,return_sequences=True))
    model.add(LSTM(units=50,dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')
    model.summary()
    return model


if __name__ =='__main__':
    data = load_dataset()
    data = class_encode(data,4)
    dataset = convert_dataset(data,n_input=n_input)
    values = dataset.values.astype('float32')

    train = values[:n_train_hours,:]
    validation = values[-n_validation_hours:,:]
    x_train,y_train = train[:,:-1],train[:,-1]
    x_validation, y_validation = validation[:, :-1], validation[:, -1]

    # data normalization
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.fit_transform((x_validation))

    # transfer data to the form [sample,time_step,feature]
    x_train = x_train.reshape(x_train.shape[0],n_input,x_train.shape[1])
    x_validation = x_validation.reshape[x_validation.shape[0],1,x_validation.shape[1]]
    # look up the dimension of data
    print(x_train.shape,y_train.shape,x_validation.shape,y_validation.shape)

    # train the model
    lstm_input_shape = (x_train.shape[1],x_train.shape[2])
    model=build_model(lstm_input_shape)
    model.fit(x_train,y_train,batch_size=batch_size,validation_data=(x_validation,y_validation),epochs=epochs,verbose=2)

    # predict
    prediction = model.predict(x_validation)

    # plot
    plt.plot(y_validation,color='blue',label='Actual')
    plt.plot(prediction,color='green',label='Prediction')
    plt.legend(loc='upper right')
    plt.show()