import numpy as np
import matplotlib.pyplot as plt

def getxy():
    t = np.arange(0,2*6.28, 2*6.28/10000)
    d = np.sin(t)
    print(t.shape)
    weier = np.zeros(10000)
    a, b = 0.94, 7
    for n in range(200):
        weier += a**n * np.cos(b**n * np.pi*t)

    #plt.plot(weier)
    #plt.show()
    N = 10000

    d = weier
    x = []
    y = []
    for i in range(N-200):

        seq = d[i:i+200]
        normaseq = (seq - np.mean(seq))/np.sqrt(np.var(seq))
        x.append(normaseq)
        if d[i+200]>d[i+199]:
            e = 1 #1 en to_categorical
        else:
            e= 0
        y.append(e)

    X = np.array(x)
    Y = np.array(y)
    print(X.shape, Y.shape)
    return X, Y
#
# if False:
#     from keras.models import Sequential
#     from keras.layers import Dense, Dropout, Activation, Flatten
#     from keras.layers import Convolution2D, MaxPooling2D
#
#     model = Sequential()
#     model.add(Dense(32, activation='relu', input_shape=(200,)))
#     model.add(Dense(2, activation='softmax'))
#
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(X, Y, batch_size=64, epochs=20)
#

