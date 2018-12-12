from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D,MaxPooling1D

seed = 7
top_words = 5000
max_words = 500
out_dimension = 32
batch_size = 128
epochs = 2
dropout_rate = 0.2

# create models
def build_model():
    model = Sequential()
    model.add(Embedding(top_words,out_dimension,input_length=max_words))
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=100))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    # output the summary information
    model.summary()
    return model

if __name__=='__main__':
    np.random.seed(seed)
    # load the data
    (x_train,y_train),(x_validation,y_validation)=imdb.load_data(num_words=top_words)

    # limit the length of dataset
    x_train = sequence.pad_sequences(x_train,maxlen=max_words)
    x_validation = sequence.pad_sequences(x_validation,maxlen=max_words)

    # create the model and train it
    model = build_model()
    model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2)
    scores = model.evaluate(x_validation,y_validation,verbose=2)
    print('Accuracy:%.2f%%'%(scores[1]*100))

