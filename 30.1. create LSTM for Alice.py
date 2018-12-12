from nltk import word_tokenize
from gensim import corpora
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D,MaxPooling1D
import numpy as np
from keras.utils import np_utils
from pyecharts import WordCloud

filename = 'Alice.txt'
document_split=['.',',','?','!',';']
batch_size = 128
epochs = 200
model_json_file = 'simple_model.json'
model_hd5_file = 'simple_model.hd5'
dict_file = 'dict_file.txt'
dict_len = 2789
max_len = 20
document_max_len = 33200

def clear_data(str):
    value = str.replace('\ufeff','').replace('\n','')
    return value

def load_dataset():
    # read the file
    with open(file=filename,mode='r') as file:
        document =[]
        lines = file.readlines()
        for line in lines:
            value = clear_data(line)
            if value != '':
                for str in word_tokenize(value):
                    if str=='CHAPTER':
                        break
                    else:
                        document.append(str.lower())
        return document

def word_to_integer(document):
    # create dictionary
    dic = corpora.Dictionary([document])
    # save dic to text
    dic.save_as_text(dict_file)
    dic_set = dic.token2id

    values=[]
    for word in document:
        values.append(dic_set[word])
    return values

def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=dict_len,output_dim=32,input_length=max_len))
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(units=dict_len,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    model.summary()

# split document according to fixed length
def make_dataset(document):
    dataset =np.array(document[0:document_max_len])
    dataset = dataset.reshape(int(document_max_len/max_len),max_len)
    return dataset

def make_y(document):
    dataset = make_dataset(document)
    y = dataset[1:dataset.shape[0],0]
    return y

def make_x(document):
    dataset = make_dataset(document)
    x = dataset[0:dataset.shape[0]-1,:]
    return x

# create wordcloud
def show_word_cloud(document):
    left_words = ['.',',','?','!',';',':','\'','(',')']
    # create dictionary
    dic = corpora.Dictionary([document])
    # calculate frequency
    words_set = dic.doc2bow(document)

    # create frequency list
    words,frequences = [],[]
    for item in words_set:
        key = item[0]
        frequence = item[1]
        word = dic.get(key=key)
        if word not in left_words:
            words.append(word)
            frequences.append(frequence)

    # use pyecharts
    word_cloud = WordCloud(width=1000,height=620)
    word_cloud.add(name='Alice\'s word cloud',attr=words,value=frequences,shape='circle',word_sizez_range=[20,100])
    word_cloud.render()

if __name__ == '__main__':
    document = load_dataset()
    show_word_cloud(document)

    # transfer words to integers
    values = word_to_integer(document)
    x_train = make_x(values)
    y_train = make_y(values)
    # one-hot encode
    y_train = np_utils.to_categorical(y_train, dict_len)

    model = build_model()
    model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2)
    # save to json file
    model_json = model.to_json()
    with open(model_json_file,'w') as file:
        file.write(model_json)
    # save weights to file
    model.save_weights(model_hd5_file)
