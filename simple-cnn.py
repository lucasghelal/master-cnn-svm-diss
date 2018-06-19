import numpy as np
from random import shuffle
from cnnmodel import cnn_model
from loadgeneric import load_dataset_bfl32
from keras import backend
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from utils import join_preds, plot_model_history, gerar_svm, write_svm, write_txt
import os


np.random.seed(123)

cnn_name = 'BFL-TrainC1-TestC3-115w-512-50ep'
nb_class = 315
imgsize = 32
batch_size = 32
nb_epochs = 50
blocos = 576

X_train, Y_train, X_test, Y_test = load_dataset_bfl32(imgsize=imgsize, nb_class=nb_class, carta_treino=1, carta_teste=3)

# nb_class = np.max(Y_train)

LUT = np.arange(len(X_train), dtype=int)
shuffle(LUT)
X_train = X_train[LUT]
Y_train = Y_train[LUT]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train[:] -= 1
Y_test[:] -= 1

Y_train = np_utils.to_categorical(Y_train, nb_class)
Y_test = np_utils.to_categorical(Y_test, nb_class)

# Rede
model = cnn_model(input_shape=(1, imgsize, imgsize), nb_class=nb_class)

if not os.path.isfile(cnn_name):
    h = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_data=(X_test, Y_test))
    plot_model_history(h.history, mode="loss", file_name=cnn_name + "-loss.png")
    plot_model_history(h.history, mode="acc", file_name=cnn_name + "-accuracy.png")
    model.save_weights(cnn_name)
else:
    model.load_weights(cnn_name)

# Combinação
Y_test_ids = [int(i/blocos) for i in range(len(Y_test))]
Y_test_class = [int(i/blocos) for i in range(len(Y_test)) if i % blocos == 0]

pred = model.predict(X_test, verbose=1)

print("")
print("SUM:", accuracy_score(Y_test_class, join_preds(pred, Y_test_ids, np.sum)))
print("PROD:", accuracy_score(Y_test_class, join_preds(pred, Y_test_ids, np.prod)))
print("MAX:", accuracy_score(Y_test_class, join_preds(pred, Y_test_ids, np.max)))



X_train, Y_train, X_test, Y_test = load_dataset_bfl32(imgsize=imgsize, nb_class=nb_class)

# nb_class = np.max(Y_train)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train[:] -= 1
Y_test[:] -= 1

Y_train = np_utils.to_categorical(Y_train, nb_class)
Y_test = np_utils.to_categorical(Y_test, nb_class)

model.summary()

# get_output = backend.function([model.layers[0].input, backend.learning_phase()], [model.layers[-3].output])

# labels_train, features_train = gerar_svm(X_train, Y_train, num_blocos=576, get_output=get_output, mode=np.average)
# labels_test, features_test = gerar_svm(X_test, Y_test, num_blocos=576, get_output=get_output, mode=np.average)
# write_svm(cnn_name + "-train.svm", features_train, labels_train)
# write_svm(cnn_name + "-test.svm", features_test, labels_test)
# write_txt(cnn_name + "-train.txt", features_train, labels_train)
