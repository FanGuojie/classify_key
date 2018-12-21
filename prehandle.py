import numpy as np
import cv2


# # 读取图片
def generate_numpy_data(start, end):
    global data, img2
    print("generate numpy data from imgs")

    def loadImg(j, m):
        global img2
        try:
            img = cv2.imread('./' + str(m + 1) + '/' + str(j)+".png" , 0)


            for i in range(2):
                img = cv2.pyrDown(img)
                ##output 120*160
            img2 = img
            data[m].append(img)
            for i in range(-1, 2):
                flipped = cv2.flip(img, i)
                data[m].append(flipped)
        except:
            # if not exist, then show last
            print(m, "kind img", j, "doesn't exist")
            data[m].append(np.array(img2))
            for i in range(-1, 2):
                flipped = cv2.flip(img2, i)
                data[m].append(flipped)

    data = [[], [], []]
    for k in range(3):
        print(k, ":")
        for i in range(start, end):
            loadImg(i, k)

        data[k] = np.array(data[k])
    data = np.array(data)
    return data


# test_data = generate_numpy_data(41,51)
# np.save("test1.npy", data)
# train_data = generate_numpy_data(1,41)
# np.save("train1.npy", data)
# data1=np.load("train1.npy")
# data2=np.load("train2.npy")
# print(data1.shape)
# print(data2.shape)

##merge dataset
def merge_dataset(filename1, filename2):
    print("merge dataset", filename1, filename2)
    data1 = np.load(filename1)
    data2 = np.load(filename2)
    ### the 2 datasets are reverse kind
    data2 = np.flipud(data2)
    data = np.concatenate((data1, data2), axis=1)
    return data


# mergeTrainData=merge_dataset("train1.npy","train2.npy")
# mergeTestData=merge_dataset("test1.npy","test2.npy")
# np.save("train.npy",mergeTrainData)
# np.save("test.npy",mergeTestData)


# 划分generate label
def generate_label(filename):
    print("generate labels from raw data & normalization from", filename)
    data = np.load(filename)
    # normalization
    data = ((data.reshape(-1) - 128) / 128).reshape(data.shape)
    rawData = []
    (_, n, l, w) = data.shape
    for i in range(3):
        # 给数据加上标注信息
        label = (np.ones(n) * (i))[:, np.newaxis]
        # 图片矩阵拉平
        rawData.append(np.concatenate((data[i].reshape(n, -1), label), axis=1))
    # 合并3类数据并shuffle
    datus = np.array(rawData).reshape(3 * n, -1)
    np.random.shuffle(datus)

    return datus, l, w


###l=120,w=160
# trainDatus, l, w = generate_label("train.npy")
# testDatus, _, _ = generate_label("test.npy")
# np.save("train_plus.npy", trainDatus)
# np.save("test_plus.npy", testDatus)

def loadfile(filename, l, w):
    datus = np.load(filename)
    data = datus[:, :-1]
    label = datus[:, -1]
    data = data.reshape(-1, l, w)
    return data, label

# trainData, trainLabel = loadfile("train_plus.npy",120,160)
# testData, testLabel = loadfile("test_plus.npy",120,160)

def quickload(filename):
    # print("quick load method")
    data = np.load(filename)
    # normalization
    data = ((data.reshape(-1) - 128) / 128).reshape(data.shape)
    rawData = []
    (_, n, l, w) = data.shape
    for i in range(3):
        # 给数据加上标注信息
        label = (np.ones(n) * (i))[:, np.newaxis]
        # 图片矩阵拉平
        rawData.append(np.concatenate((data[i].reshape(n, -1), label), axis=1))
    # 合并3类数据并shuffle
    datus = np.array(rawData).reshape(3 * n, -1)
    # np.random.shuffle(datus)
    data = datus[:, :-1]
    label = datus[:, -1]
    data = data.reshape(-1, l, w)
    return data, label

# trainData, trainLabel = quickload("train.npy")
# testData, testLabel = quickload("test.npy")
if __name__ == '__main__':
    # train=np.load("train.npy")
    #save data
    data=generate_numpy_data(1,201)
    np.save("data.npy",data)
    