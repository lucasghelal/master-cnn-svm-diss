import glob
import os
from imageio import imread
import numpy as np


# LOAD BFL
def load_data_bfl():
    data_train = []
    data_train_label = []

    base_dir = '/Users/lucas/Desktop/Base_BFL_CVL_QUWI/BFL_Textura/BFL/CF00*_[12]_*.png'

    for filename in glob.iglob(base_dir, recursive=True):
        path = filename
        label = int((os.path.split(os.path.basename(filename))[-1])[4:7])
        # 100 escritores (classes)
        if label <= 100:
            print (path)
            image = imread(path)
            image = image[:32, :32]
            image = image[np.newaxis]
            data_train.append(image)
            data_train_label.append(label)
        else:
            break

    data_train = np.array(data_train)
    data_train_label = np.array(data_train_label)

    np.save('cartasBFL-train-12.npy', data_train)
    np.save('cartasBFL-train-12_label.npy', data_train_label)


# LOAD CVL
def load_data_cvl():
    data_train = []
    data_train_label = []
    base_dir = '/Users/lucas/workspace/databases/CVL_Subset/**/*_*_00[12]_*.bmp'
    for filename in glob.iglob(base_dir, recursive=True):
        path = filename
        label = int((os.path.split(os.path.dirname(filename))[-1])[5:9])
        image = imread(path)
        image = image[:256, :256]
        image = image[np.newaxis]
        data_train.append(image)
        data_train_label.append(label)

    data_train = np.array(data_train)
    data_train_label = np.array(data_train_label)

    np.save('cartas12.npy', data_train)
    np.save('cartas12_label.npy', data_train_label)
