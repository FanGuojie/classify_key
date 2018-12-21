import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from prehandle import quickload
import torchvision
from sklearn.preprocessing import OneHotEncoder

# print("max:", np.max(train_data))
# onehot_encoder = OneHotEncoder(sparse=False)
# print(test_label.shape)
# train_label = onehot_encoder.fit_transform(train_label)
# test_label = onehot_encoder.fit_transform(test_label)

# 数据处理完成，卷积训练
torch.cuda.set_device(0)

# Hyper Parameters
EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 15
LR = 0.001  # learning rate
save_model = True


def load_train_data(filename):
    print("load train data & label")
    # (data, label) = filenames
    train_data, train_label = quickload(filename)
    train_data = train_data[:, np.newaxis]
    train_data = torch.tensor(train_data)
    train_label = torch.tensor(train_label)
    return train_data, train_label


def load_test_data(filename):
    print("load test data & label")
    # (data, label) = filenames
    test_data, test_label = quickload(filename)
    test_data = test_data[:, np.newaxis]
    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)
    return test_data, test_label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 120, 160)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 60, 80)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 60, 80)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # activation
            nn.MaxPool2d(2),  # output shape (32, 30, 40)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 30, 40)
            nn.Conv2d(32, 64, 5, 1, 2),  # output shape (64, 30, 40)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # activation
            nn.MaxPool2d(2),  # output shape (64, 15, 20)
        )
        self.conv4 = nn.Sequential(  # input shape (64, 15, 20)
            nn.Conv2d(64, 128, 5, 1, 2),  # output shape (128, 15, 20)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # activation
        )
        self.conv5 = nn.Sequential(  # input shape (128, 15, 20)
            nn.Conv2d(128, 256, 5, 1, 2),  # output shape (128, 15, 20)
            nn.BatchNorm2d(256),
            nn.Dropout(0.8),
            nn.ReLU(inplace=True),  # activation
        )
        self.conv6 = nn.Sequential(  # input shape (128, 15, 20)
            nn.Conv2d(256, 512, 5, 1, 2),  # output shape (128, 15, 20)
            nn.BatchNorm2d(512),
            nn.Dropout(0.7),
            nn.ReLU(inplace=True),  # activation
        )
        # self.conv7 = nn.Sequential(  # input shape (128, 15, 20)
        #     nn.Conv2d(512, 1024, 5, 1, 2),  # output shape (128, 15, 20)
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True),  # activation
        # )
        self.out = nn.Sequential(
            nn.Linear(512 * 15 * 20, 500),
            nn.Dropout(0.7),
            nn.Linear(500, 3),  # fully connected layer, output 7 classes
        )

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.conv2(x.float())
        x = self.conv3(x.float())
        x = self.conv4(x.float())
        x = self.conv5(x.float())
        x = self.conv6(x.float())
        # x = self.conv7(x.float())
        # for i in range(3):
        #     x=self.conv(x.float())
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


def save(model):
    print("save model")
    torch.save(model.conv1.state_dict(), 'conv1.pkl')
    torch.save(model.conv2.state_dict(), 'conv2.pkl')
    torch.save(model.conv3.state_dict(), 'conv3.pkl')
    torch.save(model.conv4.state_dict(), 'conv4.pkl')
    torch.save(model.conv5.state_dict(), 'conv5.pkl')
    torch.save(model.conv6.state_dict(), 'conv6.pkl')
    # torch.save(model.conv7.state_dict(), 'conv7.pkl')
    torch.save(model.out.state_dict(), 'out.pkl')


def restore_params(model):
    print("restore model")
    model.conv1.load_state_dict(torch.load('conv1.pkl'))
    model.conv2.load_state_dict(torch.load('conv2.pkl'))
    model.conv3.load_state_dict(torch.load('conv3.pkl'))
    model.conv4.load_state_dict(torch.load('conv4.pkl'))
    model.conv5.load_state_dict(torch.load('conv5.pkl'))
    model.conv6.load_state_dict(torch.load('conv6.pkl'))
    # model.conv7.load_state_dict(torch.load('conv7.pkl'))
    model.out.load_state_dict(torch.load('out.pkl'))


# following function (plot_with_labels) is for visualization, can be ignored if not interested
def test(model, test_loader):
    model.eval()
    acc = np.array(0)
    for step, (b_x, b_y) in enumerate(test_loader):  # gives batch data, normalize x when iterate train_loader
        b_x = b_x.cuda()
        test_output, _ = model(b_x)
        test_output = test_output.cuda()
        pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
        acc += np.sum((pred_y == b_y.data.numpy()).astype(int))
        if(not save_model):
            print(acc)
    accuracy = float(acc) / float((step+1)*test_loader.batch_size)
    return accuracy


def loadimg(path):
    print("load img")
    import cv2
    img = cv2.imread(path, 0)
    for i in range(2):
        img = cv2.pyrDown(img)
    img = ((img.reshape(-1) - 128) / 128).reshape(img.shape)[np.newaxis, np.newaxis]
    return img


def test_sample(model, data, label):
    print("test sample")
    # data=data
    test_output, _ = model(data)
    test_output = test_output.cuda()
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()[0]
    print("predict: ", pred_y)
    print("real   : ", int(label))


def init_dataloader(data, label,batch_size):
    torch_dataset = Data.TensorDataset(data, label)
    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        # shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=1,  # 多线程来读数据
    )
    return loader


def shuffleData(data, label):
    print("shuffle data")
    data=data.numpy()
    label=label.numpy()
    (n, _, l, w) = data.shape
    data = data.reshape(n,-1)
    label=label[:,np.newaxis]
    datus = np.concatenate((data, label),axis=1)
    # print(datus[:,-1][:10])
    np.random.shuffle(datus)
    data = datus[:,:-1]
    label = datus[:,-1]
    # print(label[:10])
    data = data.reshape(n, 1, l, w)
    return torch.tensor(data), torch.tensor(label)

def main():
    cnn = CNN()
    cnn.cuda()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    # training and testing
    test_data, test_label = load_test_data("test.npy")
    # test_data, test_label = shuffleData(test_data, test_label)
    if save_model:
        train_data, train_label = load_train_data("train.npy")
        # train_data, train_label = quickload("test.npy")
        # test_data, test_label = quickload("train.npy")
        # 先转换成 torch 能识别的 Dataset
        print("transfer train & test dataset , init Dataloader")
        test_loader = init_dataloader(test_data, test_label,TEST_BATCH_SIZE)
        print("start train")
        for epoch in range(EPOCH):
            if (epoch % 5 == 0):
                train_data, train_label = shuffleData(train_data, train_label)
            train_loader = init_dataloader(train_data, train_label,TRAIN_BATCH_SIZE)

            loss = 0
            for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                output = cnn(b_x)[0]  # cnn output
                loss = loss_func(output, b_y.long())  # cross entropy loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

            accuracy = test(cnn, test_loader)
            train_loss = loss.cpu().data.numpy()
            print('Epoch: ', epoch, '| train loss: %.4f' % train_loss, '| test accuracy: %.2f' % accuracy)
            # 保存参数
        save(cnn)
    else:
        # 恢复参数
        restore_params(cnn)
        test_data, test_label = load_train_data("test.npy")
        test_loader = init_dataloader(test_data, test_label,TEST_BATCH_SIZE)
        accuracy = test(cnn, test_loader)
        print("test accuracy:", accuracy)
        # sample_data = test_data[2].cpu().numpy()[np.newaxis]
        # sample_label = test_label[2].cpu().numpy()[np.newaxis]
        # sample_data = loadimg("./19_30")
        # sample_label = np.array(1)
        # sample_data = torch.tensor(sample_data).cuda()
        # test_sample(cnn, sample_data, sample_label)


if __name__ == '__main__':
    main()
    # train_data, train_label = load_train_data("train.npy")
    # train_data, train_label = shuffleData(train_data, train_label)
