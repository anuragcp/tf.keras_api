import tensorflow as tf
import numpy as np
import load_data
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

seed = 7
np.random.seed(seed=seed)

def pre_process(X):
    X = X.astype('float32')
    X = X / 255.0
    return X

def one_hot_encode(y):
    y = keras.utils.to_categorical(y=y)
    num_classes = y.shape[1]
    return y, num_classes

def define_model(num_classes, epochs):
    model = tf.keras.Sequential()
    model.add(tf.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu', kernel_constraint=keras.constraints.max_norm(3), name='input_fn'))
    model.add(tf.layers.Dropout(rate=0.2, name='drop_out1'))
    model.add(tf.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu', kernel_constraint=keras.constraints.max_norm(3), name='conv_2'))
    model.add(tf.layers.MaxPooling2D(pool_size=(2,2), strides=2, name = 'max_pool2'))
    model.add(tf.layers.Flatten(name = "flatten"))
    model.add(tf.layers.Dense(units=512, activation='relu', kernel_constraint=keras.constraints.max_norm(3), name="flatten_1"))
    model.add(tf.layers.Dropout(rate=0.5, name='drop_out2'))
    model.add(tf.layers.Dense(num_classes, activation='softmax', name = "softmax"))
    lrate = 0.01
    decay = lrate/epochs
    sgd = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #print(model.summary())
    return model

# load data
X,y=load_data.load_datasets()

# pre process
X=pre_process(X)

#one hot encode
y,num_classes=one_hot_encode(y)


#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

epochs = 10
#define model
model=define_model(num_classes,epochs)


# Fit the model
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

# list all data in history
print(history.history.keys())

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSONx
model_json = model.to_json()
with open("model_face.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_face.h5")
print("Saved model to disk")

#print("Time taken : {}".format(time.time()-t))