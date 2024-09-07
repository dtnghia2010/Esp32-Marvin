import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.data import Dataset


training_spectrogram = np.load('./training_spectrogram.npz')
validation_spectrogram = np.load('./training_spectrogram.npz')
test_spectrogram = np.load('./test_spectrogram.npz')

X_train = training_spectrogram['X']
Y_train_cats = training_spectrogram['Y']
X_validate = validation_spectrogram['X']
Y_validate_cats = validation_spectrogram['Y']
X_test = test_spectrogram['X']
Y_test_cats = test_spectrogram['Y']

IMG_WIDTH = X_train[0].shape[0]
IMG_HEIGHT = X_train[0].shape[1]

words = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
    'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
    'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
    'up', 'visual', 'wow', 'yes', 'zero', '_background'
]

Y_train = [1 if y == words.index('marvin') else 0 for y in Y_train_cats]
Y_validate = [1 if y == words.index('marvin') else 0 for y in Y_validate_cats]
Y_test = [1 if y == words.index('marvin') else 0 for y in Y_test_cats]

# Testing the Model
model2 = keras.models.load_model("./trained_model.h5")  # Use the correct .h5 extension
results = model2.evaluate(X_test, tf.cast(Y_test, tf.float32), batch_size=128)

predictions = model2.predict_on_batch(X_test)
decision = [1 if p > 0.5 else 0 for p in predictions]
tf.math.confusion_matrix(Y_test, decision)

predictions = model2.predict_on_batch(X_test)
decision = [1 if p > 0.9 else 0 for p in predictions]
tf.math.confusion_matrix(Y_test, decision)

complete_train_X = np.concatenate((X_train, X_validate, X_test))
complete_train_Y = np.concatenate((Y_train, Y_validate, Y_test))
complete_train_dataset = Dataset.from_tensor_slices((complete_train_X, complete_train_Y)).repeat(count=-1).shuffle(300000).batch(batch_size)

history = model2.fit(
    complete_train_dataset,
    steps_per_epoch=len(complete_train_X) // batch_size,
    epochs=5
)

# Final predictions and saving the fully trained model
predictions = model2.predict_on_batch(complete_train_X)
decision = [1 if p > 0.5 else 0 for p in predictions]
tf.math.confusion_matrix(complete_train_Y, decision)

decision = [1 if p > 0.95 else 0 for p in predictions]
tf.math.confusion_matrix(complete_train_Y, decision)


model2.save("./fully_trained_model.h5")