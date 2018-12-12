from keras.datasets import mnist
from matplotlib import pyplot as plt

# load data from keras
(X_train,y_train),(X_validation,y_validation) = mnist.load_data()

# show the nine images
for i in range(0,9):
    plt.subplot(331+i)
    plt.imshow(X_train[i],camp=plt.get_cmap('gray'))

plt.show()