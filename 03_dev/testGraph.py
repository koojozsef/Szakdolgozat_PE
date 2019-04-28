import numpy as np
import matplotlib.pyplot as plt


def main():
    bg_name = ["city", "nature", "sky", "tree", "waterfall"]
    fg_name = ["butterfly", "tigger1", "tigger2", "tigger3", "walk1",
               "walk2", "walk3", "walk4", "raptor", "girl", "dance"]

    dir_path = "D:/joci/projects/Szakdoga_PE/Szakdoga/Dataset/Generated/test_results/"

    data = []
    for bg in bg_name:
        for fg in fg_name:
            fname = dir_path + bg + "_" + fg + "_testlog.csv"
            f = open(fname)
            f.readline()
            f.readline()
            seq = []
            for line_n in range(100):
                line_str_lst = f.readline().split(",")
                line_array = np.array(line_str_lst, dtype=float)
                block = np.reshape(line_array[1:], (3, 4))
                seq.append(block)
            data.append(seq)
    data_array = np.array(data)
    mog_array = data_array[:, :, 0, :]
    mog2_array = data_array[:, :, 1, :]
    knn_array = data_array[:, :, 2, :]

    mog_avg = np.average(mog_array, axis=0)
    mog2_avg = np.average(mog2_array, axis=0)
    knn_avg = np.average(knn_array, axis=0)

    plt.subplot(1, 4, 1)
    plt.title("False positive (%)")
    plt.ylabel("%")
    plt.xlabel("t")
    plt.plot(mog_avg[:, 0])
    plt.plot(mog2_avg[:, 0])
    plt.plot(knn_avg[:, 0])
    plt.legend(('MoG', 'MoG2', 'KNN'),
               loc='upper right')

    plt.subplot(1, 4, 2)
    plt.title("False negative (%)")
    plt.ylabel("%")
    plt.xlabel("t")
    plt.plot(mog_avg[:, 1])
    plt.plot(mog2_avg[:, 1])
    plt.plot(knn_avg[:, 1])
    plt.legend(('MoG', 'MoG2', 'KNN'),
               loc='upper right')

    plt.subplot(1, 4, 3)
    plt.title("Kappa coefficient")
    plt.xlabel("t")
    plt.plot(mog_avg[:, 2])
    plt.plot(mog2_avg[:, 2])
    plt.plot(knn_avg[:, 2])
    plt.legend(('MoG', 'MoG2', 'KNN'),
               loc='upper right')

    plt.subplot(1, 4, 4)
    plt.title("runtime")
    plt.xlabel("t")
    plt.plot(mog_avg[:, 3])
    plt.plot(mog2_avg[:, 3])
    plt.plot(knn_avg[:, 3])
    plt.legend(('MoG', 'MoG2', 'KNN'),
               loc='upper right')

    plt.show()

    print("1")





if __name__ == "__main__":
    main()