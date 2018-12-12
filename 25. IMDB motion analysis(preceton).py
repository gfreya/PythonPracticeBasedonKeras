from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Flatten
from keras.models import Sequential

seed = 7
top_words = 5000
max_words = 500
out_dimention = 32
batch_size = 128
epochs = 2

def create_model():
    model = Sequential()
    # Embedding layers
    model.add(Embedding(top_words,out_dimention,input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

if __name__=='__main__':
    np.random.seed(seed=seed)
    # load the data
    (x_train,y_train),(x_validation,y_validation)=imdb.load_data(num_words=top_words)
    # limit the length of the dataset
    x_train = sequence.pad_sequences(x_train,maxlen=max_words)
    x_validation = sequence.pad_sequences(x_validation,maxlen=max_words)

    # create the model
    model = create_model()
    model.fit(x_train,y_train,Validation_data=(x_validation,y_validation),batch_size=batch_size,epochs=epochs,verbose=2)