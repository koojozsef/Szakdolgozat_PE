from random import random
import cv2 as cv
from math import sqrt
from keras.models import Sequential
from keras.layers import Input, Dense, regularizers, Flatten, Reshape, Conv3D
from keras.models import Model
import keras
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


def get_kernel_data(sequence_count, image_count, kernelsize, input_count, height, width):
    learn_phase = 10
    images = np.zeros((sequence_count, image_count + learn_phase, input_count, height, width)).astype(np.uint8)
    images[:, :, :2, :, :] = get_data(sequence_count, image_count + learn_phase, height, width)

    out = []
    kernel_i = int(kernelsize / 2)

    for seq in images:
        fn_mog = cv.bgsegm.createBackgroundSubtractorMOG()
        fn_knn = cv.createBackgroundSubtractorKNN()
        i = 0
        for im in seq:
            im[2] = fn_mog.apply(im[0])
            im[4] = fn_knn.apply(im[0])
            if i <= 0:
                flow = cv.calcOpticalFlowFarneback(im[0], im[0], None, 0.5, 2, 5, 3, 5, 1.2, 0)
                diff = im[0] - im[0]
                im[5] = im[2]
            else:
                flow = cv.calcOpticalFlowFarneback(seq[(i - 1), 0], im[0], None, 0.5, 2, 5, 3, 5, 1.2, 0)
                diff = np.abs(im[0] - seq[(i - 1), 0])
                im[5] = (seq[(i - 1), 2]/2 + im[2]/2).astype(np.uint8)
            # im[4] = diff.astype(np.uint8)
            a = np.einsum("ij,ij->ij", flow[:, :, 0], flow[:, :, 0])
            b = np.einsum("ij,ij->ij", flow[:, :, 1], flow[:, :, 1])
            im[3] = (cv.normalize(np.sqrt(a + b), None, 0, 255, cv.NORM_MINMAX)).astype(np.uint8)
            i = i + 1

    for seq in images[:, learn_phase:, :, :, :]:
        for im in seq:
            for k in range(kernel_i, height - kernel_i):
                for j in range(kernel_i, width - kernel_i):
                    out.append(im[:, (k - kernel_i):(k + kernel_i + 1), (j - kernel_i):(j + kernel_i + 1)])

    return np.array(out).astype(np.uint8)  # original grey, ground truth, mog, optical flow

def main():
    sequence_count = 5
    image_count = 5
    input_count = 6  # 4 input image; 0: Grey, 1: GT, 2: MOG, 3: Optical flow
    height = 120
    width = 180
    kernelshape = 96  # 5x5 kernel size

    teach_size = 300000
    test_size = 80000

    training_data_kernels = get_kernel_data(sequence_count, image_count, kernelshape, input_count, height, width)

    network_input = training_data_kernels[:, (0, 2, 3), :, :]
    network_input = np.einsum("iklm->ilmk", network_input)
    network_label = training_data_kernels[:, 1, int(kernelshape/2) + 1, int(kernelshape/2) + 1]



    #------------------
    #      keras
    #------------------

    IMG_SHAPE = (kernelshape, kernelshape, 3)
    # Create the base model from the pre-trained model MobileNet V2
    base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    base_model.trainable = False
    base_model.summary()

    model = keras.Sequential([base_model,
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dense(1, activation='sigmoid')])

    x_train = network_input[:teach_size].astype('float32') / 255.
    x_label = network_label[:teach_size].astype('float32') / 255.
    x_test = network_input[teach_size:teach_size + test_size].astype('float32') / 255.
    x_test_label = network_label[teach_size:teach_size + test_size].astype('float32') / 255.

    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    epochs = 10

    history = model.fit(x_train, x_label,
                        epochs=epochs,
                        validation_data=(x_test, x_test_label))

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.show()

    # ------------------
    #      keras - end
    # ------------------

    check_result_data = np.zeros((10, 10, input_count, height, width)).astype(np.uint8)
    check_result_data[:, :, :2, :, :] = get_data(10, 10, height, width)
    for seqi in check_result_data:
        fn_mog = cv.bgsegm.createBackgroundSubtractorMOG()
        i = 0
        for imi in seqi:
            imi[2] = fn_mog.apply(imi[0])
            if i <= 0:
                flow = cv.calcOpticalFlowFarneback(imi[0], imi[0], None, 0.5, 2, 5, 3, 5, 1.2, 0)
                diff = imi[0] - imi[0]
                imi[5] = imi[2]
            else:
                flow = cv.calcOpticalFlowFarneback(seqi[(i - 1), 0], imi[0], None, 0.5, 2, 5, 3, 5, 1.2, 0)
                diff = np.abs(imi[0] - seqi[(i - 1), 0])
                imi[5] = (seqi[(i - 1), 2] / 2 + imi[2] / 2).astype(np.uint8)
            # im[4] = diff.astype(np.uint8)
            a = np.einsum("ij,ij->ij", flow[:, :, 0], flow[:, :, 0])
            b = np.einsum("ij,ij->ij", flow[:, :, 1], flow[:, :, 1])
            imi[3] = (cv.normalize(np.sqrt(a + b), None, 0, 255, cv.NORM_MINMAX)).astype(np.uint8)
            i = i + 1

            result_im = np.zeros((height, width))
            kernel_i = int(kernelshape / 2)
            for k in range(kernel_i, height - kernelshape):
                for j in range(kernel_i, width - kernelshape):
                    pred_im = np.array([imi[(4, 2, 3), (k - kernel_i):(k + kernel_i + 1), (j - kernel_i):(j + kernel_i + 1)]])
                    pred_im = np.einsum("lijk->ljki", pred_im)
                    result_im[k, j] = model.predict(pred_im)

            print(np.max(result_im))
            print(np.sum(result_im))
            res_im = cv.normalize(result_im, None, 0, 255, cv.NORM_MINMAX)
            cv.imshow("NN 0", res_im.astype(np.uint8))
            cv.imshow("flow", imi[3])
            cv.imshow("mog", imi[5])
            cv.imshow("image", imi[0])
            cv.imshow("gt", imi[1])
            cv.waitKey()


if __name__ == '__main__':
    main()
