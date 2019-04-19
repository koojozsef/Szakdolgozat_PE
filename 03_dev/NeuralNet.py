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


def get_kernel_data(sequence_count, image_count, kernelsize, height, width):
    learn_phase = 10
    images = np.zeros((sequence_count, image_count + learn_phase, 4, height, width)).astype(np.uint8)
    images[:, :, :2, :, :] = get_data(sequence_count, image_count + learn_phase, height, width)

    out = []
    kernel_i = int(kernelsize / 2)

    for seq in images:
        fn_mog = cv.bgsegm.createBackgroundSubtractorMOG()
        i = 0
        for im in seq:
            im[2] = fn_mog.apply(im[0])
            if i <= 0:
                flow = cv.calcOpticalFlowFarneback(im[0], im[0], None, 0.5, 2, 5, 3, 5, 1.2, 0)
            else:
                flow = cv.calcOpticalFlowFarneback(seq[(i - 1), 0], im[0], None, 0.5, 2, 5, 3, 5, 1.2, 0)
            a = np.einsum("ij,ij->ij", flow[:, :, 0], flow[:, :, 0])
            b = np.einsum("ij,ij->ij", flow[:, :, 1], flow[:, :, 1])
            im[3] = (cv.normalize(np.sqrt(a + b), None, 0, 255, cv.NORM_MINMAX)).astype(np.uint8)
            i = i + 1

    for seq in images[:, learn_phase:, :, :, :]:
        for im in seq:
            for k in range(kernel_i, height - kernelsize):
                for j in range(kernel_i, width - kernelsize):
                    out.append(im[:, (k - kernel_i):(k + kernel_i + 1), (j - kernel_i):(j + kernel_i + 1)])

    return np.array(out).astype(np.uint8)  # original grey, ground truth, mog, optical flow

def main():
    sequence_count = 20
    image_count = 5
    input_count = 4  # 4 input image; 0: Grey, 1: GT, 2: MOG, 3: Optical flow
    height = 160
    width = 240
    kernelshape = 15  # 5x5 kernel size

    teach_size = 10000
    test_size = 2000


    training_data_kernels = get_kernel_data(sequence_count, image_count, kernelshape, height, width)

    network_input = training_data_kernels[:, (0, 2, 3), :, :]
    network_label = training_data_kernels[:, 1, int(kernelshape/2) + 1, int(kernelshape/2) + 1]



    #------------------
    #      keras
    #------------------

    input_img = Input(shape=(3, kernelshape, kernelshape))

    encoded1 = Dense(25, activation='relu',
                     activity_regularizer=regularizers.l2(10e-3))(input_img)
    encoded2 = Dense(20, activation='relu',
                     activity_regularizer=regularizers.l2(10e-3))(encoded1)
    encoded3 = Dense(5, activation='relu',
                     activity_regularizer=regularizers.l1(10e-3))(encoded2)

    flatten = Flatten()(encoded3)

    decoded = Dense(1, activation='sigmoid')(flatten)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    #build
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    x_train = network_input[:teach_size].astype('float32') / 255.
    x_label = network_label[:teach_size].astype('float32') / 255.
    x_test = network_input[teach_size:teach_size + test_size].astype('float32') / 255.
    x_test_label = network_label[teach_size:teach_size + test_size].astype('float32') / 255.

    autoencoder.fit(x_train, x_label,
                    epochs=3,
                    batch_size=500,
                    shuffle=True,
                    validation_data=(x_test, x_test_label))

    # ------------------
    #      keras - end
    # ------------------

    check_result_data = np.zeros((10, 10, input_count, height, width)).astype(np.uint8)
    check_result_data[:, :, :2, :, :] = get_data(10, 10, height, width)
    for seqi in check_result_data:
        fn_mog = cv.bgsegm.createBackgroundSubtractorMOG()
        i = 0
        for imi in seqi:
            mog_res = fn_mog.apply(imi[0])
            if i <= 0:
                flow = cv.calcOpticalFlowFarneback(imi[0], imi[0], None, 0.5, 2, 5, 3, 5, 1.2, 0)
            else:
                flow = cv.calcOpticalFlowFarneback(seqi[(i - 1), 0], imi[0], None, 0.5, 2, 5, 3, 5, 1.2, 0)
            a = np.einsum("ij,ij->ij", flow[:, :, 0], flow[:, :, 0])
            b = np.einsum("ij,ij->ij", flow[:, :, 1], flow[:, :, 1])
            imi[3] = (cv.normalize(np.sqrt(a + b), None, 0, 255, cv.NORM_MINMAX)).astype(np.uint8)
            i = i + 1

            result_im = np.zeros((height, width))
            kernel_i = int(kernelshape / 2)
            for k in range(kernel_i, height - kernelshape):
                for j in range(kernel_i, width - kernelshape):
                    pred_im = np.array([imi[(0, 2, 3), (k - kernel_i):(k + kernel_i + 1), (j - kernel_i):(j + kernel_i + 1)]])
                    result_im[k, j] = autoencoder.predict(pred_im)

            print(np.max(result_im))
            print(np.sum(result_im))
            res_im = cv.normalize(result_im, None, 0, 255, cv.NORM_MINMAX)
            cv.imshow("NN 0", res_im.astype(np.uint8))
            cv.imshow("flow", imi[3])
            cv.imshow("mog", mog_res)
            cv.imshow("image", imi[0])
            cv.imshow("gt", imi[1])
            cv.waitKey()


if __name__ == '__main__':
    main()
