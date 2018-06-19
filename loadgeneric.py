import glob
import os
from imageio import imread
import numpy as np
import hashlib


def load_dataset_bfl(imgsize=32, nb_class=115):
    X_train, Y_train = load_dataset('/Users/lucas/Desktop/mestrado/Base_BFL_CVL_QUWI/BFL_Textura/BFL/CF00*_[1]_*.png', start=4, end=7, imgsize=imgsize, maxlabel=nb_class)

    X_test, Y_test = load_dataset('/Users/lucas/Desktop/mestrado/Base_BFL_CVL_QUWI/BFL_Textura/BFL/CF00*_[3]_*.png', start=4, end=7, imgsize=imgsize, maxlabel=nb_class)

    return X_train, Y_train, X_test, Y_test

def load_dataset_bfl32(imgsize=32, nb_class=115, carta_treino=1, carta_teste=2):
    X_train, Y_train = load_dataset('/Users/lucas/Desktop/bfl32/CF00*_['+ str(carta_treino)+']_*.png', start=4, end=7, imgsize=imgsize, maxlabel=nb_class)

    X_test, Y_test = load_dataset('/Users/lucas/Desktop/bfl32/CF00*_['+ str(carta_teste)+']_*.png', start=4, end=7, imgsize=imgsize, maxlabel=nb_class)

    return X_train, Y_train, X_test, Y_test


def load_dataset_cvl(basedir='/Users/lucas/workspace/databases/CVL_Subset/**/*_*_00[12]_*.bmp', imgsize=256):
    return load_dataset(basedir, start=5, end=9, imgsize=imgsize)


def load_dataset(basedir, imgsize=128, start=0, end=0, maxlabel=None):
    # carrega se existir no cache (.npy)
    hashstr = "|".join([str(basedir), str(imgsize), str(start), str(end), str(maxlabel)]).encode('utf-8')
    hash = hashlib.sha1(hashstr).hexdigest()

    datafile = hash + "_data.npy"
    labelfile = hash + "_label.npy"

    if os.path.isfile(datafile) and os.path.isfile(labelfile):
        return np.load(datafile), np.load(labelfile)

    data = []
    labels = []

    for filepath in glob.iglob(basedir, recursive=True):
        filename = os.path.basename(filepath)
        print('Carregando %s' % (filename))

        label = int(filename[start:end])

        if maxlabel and label > maxlabel:
            break

        image = imread(filepath)
        image = image[:imgsize, :imgsize]
        image = image[np.newaxis]

        data.append(image)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    np.save(datafile, data)
    np.save(labelfile, labels)

    return data, labels
