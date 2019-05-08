from random import random
import cv2 as cv
from math import sqrt
from keras.models import Sequential
from keras.layers import Input, Dense, regularizers, Flatten, Reshape, Conv3D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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


def get_random_image_sequence(src_array, number_of_image, height, width):
    sequence = int(random() * src_array.shape[0])
    max_image = len(os.listdir(src_array[sequence, 0]))
    image_start = int(random() * max((max_image - number_of_image - 1), 0))
    ret = []
    for i in range(number_of_image):
        path = str(src_array[sequence, 0]) + "%04d" % (i + image_start) + ".png"
        path_gt = str(src_array[sequence, 1]) + "%04d" % (i + image_start) + ".png"
        image = cv.imread(path, 0)
        image_gt = cv.imread(path_gt, 0)
        image = cv.resize(image, (width, height))
        image_gt = cv.resize(image_gt, (width, height))
        ret.append([image, image_gt])

    return ret


def get_data(seq_count, im_count, height, width):
    images_source = get_test_data_source()
    sequences = []
    for i in range(seq_count):
        image_array = get_random_image_sequence(images_source, im_count, height, width)
        sequences.append(image_array)
    return np.asarray(sequences).astype(np.uint8)


def main():
    sequence_count = 20
    image_count = 5
    input_count = 4  # 4 input image; 0: Grey, 1: GT, 2: MOG, 3: Optical flow
    height = 80
    width = 120

    training_data = np.zeros((sequence_count, image_count, input_count, height, width)).astype(np.uint8)
    training_data[:, :, :2, :, :] = get_data(sequence_count, image_count, height, width)

    for seq in training_data:
        fn_mog = cv.bgsegm.createBackgroundSubtractorMOG()
        i = 0
        for im in seq:
            im[2] = fn_mog.apply(im[0])
            if i <= 0:
                flow = cv.calcOpticalFlowFarneback(im[0], im[0], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            else:
                flow = cv.calcOpticalFlowFarneback(seq[(i - 1), 0], im[0], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            a = np.einsum("ij,ij->ij", flow[:, :, 0], flow[:, :, 0])
            b = np.einsum("ij,ij->ij", flow[:, :, 1], flow[:, :, 1])
            im[3] = (cv.normalize(np.sqrt(a + b), None, 0, 255, cv.NORM_MINMAX)).astype(np.uint8)
            i = i + 1

    network_input = np.reshape(np.array(training_data[:, :, (0, 2, 3), :, :]),
                               (sequence_count * image_count, 3, height, width))
    network_label = np.reshape(np.array(training_data[:, :, 1, :, :]),
                               (sequence_count * image_count, height, width))


    #------------------
    #      keras
    #------------------

    input_img = Input(shape=(3, height, width))

    encoded1 = Dense(80, activation='relu',
                    activity_regularizer=regularizers.l1(10e-3))(input_img)
    encoded2 = Dense(80, activation='relu',
                    activity_regularizer=regularizers.l1(10e-3))(encoded1)
    encoded3 = Dense(80, activation='relu',
                    activity_regularizer=regularizers.l1(10e-3))(encoded2)

    flatten = Flatten()(encoded3)

    decoded = Dense(height*width, activation='sigmoid')(flatten)

    reshape = Reshape((height, width))(decoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, reshape)

    #build
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    x_train = network_input[:70].astype('float32') / 255.
    x_label = network_label[:70].astype('float32') / 255.
    x_test = network_input[70:].astype('float32') / 255.
    x_test_label = network_label[70:].astype('float32') / 255.

    autoencoder.fit(x_train, x_label,
                    epochs=15,
                    batch_size=2,
                    shuffle=True,
                    validation_data=(x_test, x_test_label))

    check_result_data = np.zeros((2, 10, input_count, height, width)).astype(np.uint8)
    check_result_data[:, :, :2, :, :] = get_data(2, 10, height, width)
    for seqi in check_result_data:
        fn_mog = cv.bgsegm.createBackgroundSubtractorMOG()
        fn_mog2 = cv.createBackgroundSubtractorMOG2(detectShadows=0)
        fn_knn = cv.createBackgroundSubtractorKNN(detectShadows=0)
        i = 0
        for imi in seqi:
            mog2_res = fn_mog2.apply(imi[0])
            knn_res = fn_knn.apply(imi[0])
            mog_res = fn_mog.apply(imi[0])
            if i <= 0:
                flow = cv.calcOpticalFlowFarneback(imi[0], imi[0], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            else:
                flow = cv.calcOpticalFlowFarneback(seqi[(i - 1), 0], imi[0], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            a = np.einsum("ij,ij->ij", flow[:, :, 0], flow[:, :, 0])
            b = np.einsum("ij,ij->ij", flow[:, :, 1], flow[:, :, 1])
            imi[3] = (cv.normalize(np.sqrt(a + b), None, 0, 255, cv.NORM_MINMAX)).astype(np.uint8)
            i = i + 1
            pred_im = np.array([imi[(0, 2, 3), :, :]])
            result_im = autoencoder.predict(pred_im)
            print(np.max(result_im))
            print(np.sum(result_im))
            cv.imshow("NN 0", (result_im[0]*255).astype(np.uint8))
            cv.imshow("flow", imi[3])
            cv.imshow("mog", mog_res)
            cv.imshow("mog2", mog2_res)
            cv.imshow("knn", knn_res)
            cv.imshow("image", imi[0])
            cv.imshow("gt", imi[1])
            cv.waitKey()


if __name__ == '__main__':
    main()
