from imageio import imread, imwrite
import glob
import os
import numpy as np

def split(img, label, ratioY=4, ratioX=4):
    new_size = ratioX*ratioY
    new_size_x = int(img.shape[1]/ratioX)
    new_size_y = int(img.shape[2]/ratioY)

    X_new = np.zeros((new_size, 1, new_size_x, new_size_y))
    Y_new = np.zeros(new_size)

    pos = 0
    for row in range(ratioX):
        for col in range(ratioY):
            X_new[pos][0] = img[0][new_size_x*row:new_size_x*(row+1),
                                   new_size_y*col:new_size_y*(col+1)]
            Y_new[pos] = label
            pos += 1
    
    assert(pos == new_size)

    return X_new, Y_new

def split_files(path, save_path, ratioX, ratioY, expected_size=None):
    for file_path in glob.iglob(path, recursive=True):
        filename = os.path.basename(file_path)
        print('loading %s' % filename)

        image = imread(file_path)
        image = image.T
        image = image[np.newaxis]

        # inverter ratio por causa que é a transposta da imagem
        imagens, labels = split(image, 0, ratioY=ratioY, ratioX=ratioX)

        if expected_size:
            assert(expected_size == imagens[0][0].shape)

        for i, image_new in enumerate(imagens):
            imwrite(os.path.join(save_path, "%s_%d.png" % (filename[:-4], i)), np.array(image_new[0], dtype="uint8").T) 


path = '/Users/lucas/Desktop/mestrado/Bases - Lucas/Base_BFL_CVL_QUWI/BFL_Textura/*'
save_path = '/Users/lucas/Desktop/bfl32/'

split_files(path, save_path, 72, 8, expected_size=(32, 32))
