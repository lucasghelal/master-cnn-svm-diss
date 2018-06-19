from keras.utils import np_utils
from keras import backend
from cnnmodel import cnn_model
import numpy as np
from utils import join_preds, plot_model_history, gerar_svm, write_svm, write_txt

nb_class = 315
imgsize = 32
cnn_name = 'BFL-TrainC2-TestC3_DISS_3'
cnn_name_weights = 'BFL-TrainC1-TestC3-115w-512-50ep'

model = cnn_model(input_shape=(1, imgsize, imgsize), nb_class=nb_class)
model.load_weights(cnn_name_weights)

X_train, Y_train, X_test, Y_test = load_dataset_bfl32(imgsize=imgsize, nb_class=nb_class, carta_treino=2, carta_teste=3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train[:] -= 1
Y_test[:] -= 1

Y_train = np_utils.to_categorical(Y_train, nb_class)
Y_test = np_utils.to_categorical(Y_test, nb_class)


get_output = backend.function([model.layers[0].input, backend.learning_phase()], [model.layers[-3].output])

labels_train, features_train = gerar_svm(X_train, Y_train, num_blocos=192, get_output=get_output, mode=np.average)

labels_test, features_test = gerar_svm(X_test, Y_test, num_blocos=192, get_output=get_output, mode=np.average)
# write_svm(cnn_name + "-train.svm", features_train, labels_train)
# write_svm(cnn_name + "-test.svm", features_test, labels_test)
write_txt(cnn_name + "-train.txt", features_train, labels_train)
write_txt(cnn_name + "-test.txt", features_test, labels_test)

