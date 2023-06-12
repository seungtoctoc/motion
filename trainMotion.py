import numpy as np
import os
import sys

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# parameter (docName)
docName = sys.argv[1]


# test
# docName = "final2"


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# make folder 'models'
os.makedirs('models', exist_ok=True)


actions = [
    'first',
    'second',
    'third'
]


# load sequence data
data = np.concatenate([
    np.load("dataset/seq_{}_first.npy".format(docName)),
    np.load("dataset/seq_{}_second.npy".format(docName)),
    np.load("dataset/seq_{}_third.npy".format(docName))
], axis=0)


# [data preprocessing]

# value excluding label
x_data = data[:, :, :-1]


# assign label value
labels = data[:, 0, -1]
# assign integer values to labels
label_dict = {label: i for i, label in enumerate(actions)}
# convert to integer
labels = np.array([label_dict[label] for label in labels])


# one-hot encoding
y_data = to_categorical(labels, num_classes=len(actions))


# convert data type
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)


# training set 90%, test set 10%
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)


# [model]

# create Sequential model
# connet LSTM and two Dense
model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])


# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


# learning 
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/model_{}.h5'.format(docName), monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)