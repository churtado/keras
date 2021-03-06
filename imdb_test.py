from keras.datasets import imdb
from keras import models
from keras import layers
from keras import regularizers

import matplotlib.pyplot as plt

# only top 10k most frequent words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # all zero matrix of that shape
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # set the positions to 1 that need to be set
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# this was already one-hot encoded into 0-1 (binary), so not much work
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


'''
since the format of the data is simple, densely connected layers work well with relu activation functions
'''

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(.001), activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(.001), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid')) # probability of a positive review

'''
We're going to be using binary cross-entropy for a loss function because it's usually the best one for this type of 
network and data. This function is best when dealing with probabilities. You'll have to research what the heck it means
'''

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# let's create a validation data set

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))

#print(history.history)
#print(history.history.keys())

history_dict = history.history

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

print(model.predict(x_test))

'''
What we can see with the original values is a case of overfitting: excellent on training
bad on validation. It's not generalizing well. So we change to 4 epochs
'''