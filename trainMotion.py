import numpy as np
import os
import sys

# parameter (docName)
docName = sys.argv[1]


# docName = "testDoc3"

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions = [
    'first',
    'second',
    'third'
]

# load seq data
data = np.concatenate([
    np.load("dataset/seq_{}_first.npy".format(docName)),
    np.load("dataset/seq_{}_second.npy".format(docName)),
    np.load("dataset/seq_{}_third.npy".format(docName))
], axis=0)

# print data shape
# data.shape

# make x data witout last value(label)
x_data = data[:, :, :-1]

# make label with last value
labels = data[:, 0, -1]

label_dict = {label: i for i, label in enumerate(actions)}
labels = np.array([label_dict[label] for label in labels])

# print(x_data.shape)
# print(labels.shape)



from tensorflow.keras.utils import to_categorical

# one-hot encoding
y_data = to_categorical(labels, num_classes=len(actions))
# y_data.shape



from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

# 90퍼 -> 트레이닝 셋, 10퍼 -> 테스트셋으로 만듦
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)


# model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# connet LSTM and two Dense
model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# model.summary()



# learning, 이 과정에서 오래 걸리는 듯
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# learning with model.fit, 200 epochs. save accurate model
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



# graph
# green, blue -> train acc, val acc
# yellow, red -> loss
# import matplotlib.pyplot as plt

# fig, loss_ax = plt.subplots(figsize=(16, 10))
# acc_ax = loss_ax.twinx()

# loss_ax.plot(history.history['loss'], 'y', label='train loss')
# loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper left')

# acc_ax.plot(history.history['acc'], 'b', label='train acc')
# acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='upper left')

# plt.show()



# from sklearn.metrics import multilabel_confusion_matrix
# from tensorflow.keras.models import load_model

# # load saved model
# model = load_model('models/model_{}.h5'.format(docName))

# y_pred = model.predict(x_val)

# multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))