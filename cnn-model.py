from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend

backend.set_image_data_format('channels_first')


def cnn_model(input_shape, nb_class):
    input_conv = Input(shape=input_shape)
    conv = Conv2D(96, (3, 3), padding="same")(input_conv)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv)
    x = Activation('relu')(x)

    x = Conv2D(128, (6, 6), padding="same")(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Activation('relu')(x)

    x = Flatten()(x)

    x = Dense(4096, activation="relu")(x)
    x = Activation('relu')(x)

    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu")(x)
    x = Activation('relu')(x)

    output = Dense(nb_class)(x)
    output = Activation('softmax')(output)

    model = Model(inputs=input_conv, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    return model
