import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import quantize_resnet
import keras
import random
from keras.utils import np_utils
import h5py
from layercnn import cnn_10layer
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping,ModelCheckpoint
train_path = 'F:/diabetic-retinopathy-detection/flow/train/'
test_path = 'F:/diabetic-retinopathy-detection/flow/train_val/'

img_rows, img_cols = 256,256



batch_size = 128
nb_classes = 2
epochs = 100

from sklearn.utils import shuffle

f1 = h5py.File('x_train.h5', 'r')
x_train = f1['dataset_1']

f2 = h5py.File('y_train.h5', 'r')
y_train = f2['dataset_1']

f3 = h5py.File('x_test.h5', 'r')
x_test = f3['dataset_1']

f4 = h5py.File('y_test.h5', 'r')
y_test = f4['dataset_1']
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train, y_train = shuffle(x_train, y_train)

path = 'weight_quan'+'/'
if not os.path.exists(path):
    os.mkdir(path)
checkpoint = ModelCheckpoint((path+'resnet18.{epoch:02d}-{val_acc:.2f}.hdf5'), verbose=1, monitor='val_acc',save_best_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger(path+'resnet18_PPG.csv')
model = quantize_resnet.ResnetBuilder.build_resnet_18((3, img_rows, img_cols), nb_classes)

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=2,
      validation_split=0.2,
      shuffle=True,
      class_weight='auto',
      callbacks=[csv_logger,checkpoint])


f = open(path + '/result.csv', 'w')
for file in os.listdir(path):
    if 'hdf5' in file:
        print(file)
        f.write(file + '\n')
        model.load_weights(path + '/' + file)
        y_pred = np.argmax(model.predict(x_test), axis=-1)
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

        print(accuracy_score(y_test, y_pred))
        f.write('Accurary: ' + str(accuracy_score(y_test, y_pred)) + '\n')
        c = confusion_matrix(y_test, y_pred)
        print('Confusion matrix:\n', c)
        f.write('Confusion matrix:\n' + str(c))
        print(classification_report(y_test, y_pred))
        f.write('---------------------------------------------------------------------------------------\n')
f.close()