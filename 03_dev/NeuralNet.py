from random import random
import cv2 as cv
from keras.models import Sequential
from keras.layers import Input, Dense, regularizers
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


def get_test_data_source():
    bg_name = ["city", "nature", "sky", "tree", "waterfall"]
    fg_name_normal = ["butterfly", "walk1", "walk2", "walk3", "walk4"]

    ret_data = []

    for bg in bg_name:
        for fg in fg_name_normal:
            orig_dest = str(bg + "_" + fg)
            raw_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + orig_dest + "\\orig\\"
            mask_folder = "D:\\joci\\projects\\Szakdoga_PE\\Szakdoga\\Dataset\\Generated\\database\\" + orig_dest + "\\gt\\"
            ret_data.append([raw_folder, mask_folder])

    return np.asarray(ret_data)


def get_random_image_sequence(src_array, number_of_image):
    sequence = int(random() * src_array.shape[0])
    max_image = len(os.listdir(src_array[sequence, 0]))
    image_start = int(random() * max((max_image - number_of_image - 1), 0))
    ret = []
    for i in range(number_of_image):
        path = str(src_array[sequence, 0]) + "%04d" % (i + image_start) + ".png"
        path_gt = str(src_array[sequence, 1]) + "%04d" % (i + image_start) + ".png"
        image = cv.imread(path)
        image_gt = cv.imread(path_gt)
        image = cv.resize(image, (240, 160))
        image_gt = cv.resize(image_gt, (240, 160))
        ret.append([image, image_gt])

    return ret


def get_data(seq_count, im_count):
    images_source = get_test_data_source()
    sequences = []
    for i in range(seq_count):
        image_array = get_random_image_sequence(images_source, im_count)
        sequences.append(image_array)
    return sequences


def main():
    sequence_count = 40
    image_count = 80
    training_data = get_data(sequence_count, image_count)

    #------------------
    #      keras
    #------------------

    input_img = Input(shape=(imageshape,))
    # add a Dense layer with a L1 activity regularizer
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(input_img)
    decoded = Dense(imageshape, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    #build
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    x_train = images[:1000].astype('float32') / 255.
    x_test = images[9000:].astype('float32') / 255.

    autoencoder.fit(x_train, x_train,
                    epochs=1,
                    batch_size=200,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    # use Matplotlib (don't ask)
    import matplotlib.pyplot as plt

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(np.transpose(x_test[i].reshape(3,32,32),(1,2,0)))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.transpose(decoded_imgs[i].reshape(3,32,32),(1,2,0)))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()
