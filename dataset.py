# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2023年09月18日
"""
import numpy as np
import scipy.io as sio
import os
import torch
import torch.utils.data as Data
import random
from sklearn.decomposition import PCA
from einops import rearrange
from module import get_image_cubes


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def load_data(dataset):
    data_path = r'D:\Program Files (x86)\Anaconda\jupyter_path\dataset'
    if dataset == 'Houston': #HSI.shape (349, 1905, 144), LiDAR.shape (349, 1905) gt.shape (349, 1905)
        HSI_data = sio.loadmat(os.path.join(data_path, 'Houston2013/HSI.mat'))['HSI'] # (349, 1905, 144)
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'Houston2013/LiDAR.mat'))['LiDAR'] # (349, 1905)
        LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
        Train_data = sio.loadmat(os.path.join(data_path, 'Houston2013/TRLabel.mat'))['TRLabel'] # (349, 1905)
        Test_data = sio.loadmat(os.path.join(data_path, 'Houston2013/TSLabel.mat'))['TSLabel'] # (349, 1905)
        GT = sio.loadmat(os.path.join(data_path, 'Houston2013/gt.mat'))['gt'] # (349, 1905)

    if dataset == 'Berlin':
        HSI_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/data_HS_LR.mat'))['data_HS_LR'] #(1723, 476, 244) 8类
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/data_SAR_HR.mat'))['data_SAR_HR']
        Train_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/TrainImage.mat'))['TrainImage']
        Test_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/TestImage.mat'))['TestImage']
        GT = Train_data + Test_data

    if dataset == 'Trento':
        HSI_data = sio.loadmat(os.path.join(data_path, 'Trento/HSI.mat'))['HSI'] #HSI.shape
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'Trento/LiDAR.mat'))['LiDAR'] #LiDAR.shape
        LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
        Train_data = sio.loadmat(os.path.join(data_path, 'Trento/TRLabel.mat'))['TRLabel']
        Test_data = sio.loadmat(os.path.join(data_path, 'Trento/TSLabel.mat'))['TSLabel']
        GT = sio.loadmat(os.path.join(data_path, 'Trento/gt.mat'))['gt']

    if dataset == 'MUUFL':
        HSI_data = sio.loadmat(os.path.join(data_path, 'MUUFL/HSI.mat'))['HSI'] #
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'MUUFL/LiDAR.mat'))['LiDAR']
        Train_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_train_150.mat'))['mask_train']
        Test_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_test_150.mat'))['mask_test']
        GT = sio.loadmat(os.path.join(data_path, 'MUUFL/gt.mat'))['gt']
        GT[GT==-1] = 0

    if dataset == 'Augsburg':
        HSI_data = sio.loadmat(os.path.join(data_path, 'Augsburg/data_HS_LR.mat'))['data_HS_LR']#(332, 485, 180)
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'Augsburg/data_DSM.mat'))['data_DSM']
        LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
        Train_data = sio.loadmat(os.path.join(data_path, 'Augsburg/TrainImage.mat'))['TrainImage']
        Test_data = sio.loadmat(os.path.join(data_path, 'Augsburg/TestImage.mat'))['TestImage']
        GT = Train_data + Test_data

    return HSI_data, LiDAR_data, Train_data, Test_data, GT


def shape_new_X(x):
    x[:, 1::2, :] = np.flip(x[:, 1::2, :], axis=[-1])
    return x

def neighbour_band1(x, patch):  # x (32, 169, 144)
    x = rearrange(x, 'n c h w -> n (h w) c')
    (b, n, d) = x.shape
    x = shape_new_X(x)
    x_new = np.zeros((b, n+2*patch, d)) #(32, 169+6, 144)
    x_new[:, patch:patch+n, : ] = x
    x_new[:, :patch, :] = np.flip(x[:, :patch, :], axis=[-1])
    x_new[:, patch+n:, :] = np.flip(x[:, n-patch:, :], axis=[-1])
    # x_new_rev = np.flip(x_new, axis=[-2, -1])
    x1 = np.zeros((b, n, patch, d), dtype=float)
    for i in range(n):
        x1[:, i, :, :] = x_new[:, i:i + patch, :]
    out = rearrange(x1, 'b n p c -> b (n p) c')
    return out

def neighbour_band2(x, patch):  # x (32, 169, 144)
    x = rearrange(x, 'n c h w -> n c w h')
    x = rearrange(x, 'n c w h -> n (w h) c')
    (b, n, d) = x.shape
    x = shape_new_X(x)
    x_new = np.zeros((b, n+2*patch, d)) #(32, 169+6, 144)
    x_new[:, patch:patch+n, : ] = x
    x_new[:, :patch, :] = np.flip(x[:, :patch, :], axis=[-1])
    x_new[:, patch+n:, :] = np.flip(x[:, n-patch:, :], axis=[-1])
    # x_new_rev = np.flip(x_new, axis=[-2, -1])
    x1 = np.zeros((b, n, patch, d), dtype=float)
    for i in range(n):
        x1[:, i, :, :] = x_new[:, i:i + patch, :]
    out = rearrange(x1, 'b n p c -> b (n p) c')
    return out

def pixel_select(Y, train_ratio):
    """
    :param Y:
    :param train_ratio: 训练集比率
    :return:
    """
    test_pixels = Y.copy()  # 复制Y到test_pixels
    kinds = np.unique(Y).shape[0] - 1  # np.unique(Y)=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16],dtype=uint8) ,kinds=分类种类数
    # print(kinds)
    for i in range(kinds):
        num = np.sum(Y == (i + 1))  # 计算每个类总共有多少样本 ,从Y=1到Y=16
        train_num = int(round(num * train_ratio))
        temp1 = np.where(Y == (i + 1))  # 返回标签满足第i+1类的位置索引，第一次循环返回第一类的索引
        temp2 = random.sample(range(num), train_num)  # get random sequence,random.sample表示从某一序列中随机获取所需个数（train_num）的数并以片段的形式输出,,再这里将随机从每个种类中挑选train_num个样本
        for i in temp2:
            test_pixels[temp1[0][temp2], temp1[1][temp2]] = 0  # 除去训练集样本

    train_pixels = Y - test_pixels
    return train_pixels, test_pixels

def pixel_selection(gt, train_pixel):
    test_pixels = gt.copy()
    kinds = np.unique(gt).shape[0]-1
    for i in range(kinds):
        num = np.sum(train_pixel==(i+1))
        val_num = int(num * 0.9)
        temp1 = np.where(train_pixel==(i+1))
        temp2 = random.sample(range(num), val_num)
        for i in temp2:
            test_pixels[temp1[0][temp2], temp1[1][temp2]] = 0  # 除去训练集样本
    train_pixels = gt - test_pixels
    return train_pixels, test_pixels


def generater(X_hsi, X_lidar, train_pixels, test_pixels, GT, batch_size, windowSize):
    x_train_hsi, y_train_hsi = get_image_cubes(input_data=X_hsi, pixels_select=train_pixels, windowSize=windowSize)
    x_test_hsi, y_test_hsi = get_image_cubes(input_data=X_hsi, pixels_select=test_pixels, windowSize=windowSize)

    x_train_lidar, y_train_lidar = get_image_cubes(input_data=X_lidar, pixels_select=train_pixels, windowSize=windowSize)
    x_test_lidar, y_test_lidar = get_image_cubes(input_data=X_lidar, pixels_select=test_pixels, windowSize=windowSize)


    TRAIN_SIZE = x_train_hsi.shape[0]
    TEST_SIZE = x_test_hsi.shape[0]
    TOTAL_SIZE = x_train_hsi.shape[0] + x_test_hsi.shape[0]

    print('X_train:{}\nX_test:{}\nX_all:{}'.format(x_train_hsi.shape[0], x_test_hsi.shape[0], x_train_hsi.shape[0]+x_test_hsi.shape[0]))

    hsi_train_tensor = torch.from_numpy(x_train_hsi).type(torch.FloatTensor)
    hsi_test_tensor = torch.from_numpy(x_test_hsi).type(torch.FloatTensor)

    lidar_train_tensor = torch.from_numpy(x_train_lidar).type(torch.FloatTensor)
    lidar_test_tensor = torch.from_numpy(x_test_lidar).type(torch.FloatTensor)

    y_train = torch.from_numpy(y_train_hsi).type(torch.int64)
    y_test = torch.from_numpy(y_test_hsi).type(torch.int64)

    torch_train = Data.TensorDataset(hsi_train_tensor, lidar_train_tensor, y_train)
    torch_test = Data.TensorDataset(hsi_test_tensor, lidar_test_tensor, y_test)

    train_iter = Data.DataLoader(
        dataset=torch_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_iter = Data.DataLoader(
        dataset=torch_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter

def normalize(X, type):
    x = np.zeros(shape=X.shape, dtype='float32')
    if type == 1:
        for i in range(X.shape[2]):
            temp = X[:, :, i]
            mean = np.mean(temp)
            std = np.std(temp)
            x[:, :, i] = ((temp - mean) / std)
    if type == 2:
        for i in range(X.shape[2]):
            x[:, :, i] = (X[:, :, i]-X[:, :, i].min()) / (X[:, :, i].max() - X[:, :, i].min())
    return x

