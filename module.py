# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2023年09月18日
"""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
import torch
from einops import rearrange
from operator import truediv
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from munkres import Munkres
from sklearn import metrics

def result(test_iter, dataset, device, net):
    net.load_state_dict(torch.load('./models/' + dataset + '.pt'))
    print('\n***Start  Testing***\n')
    y_test = []
    y_pred = []
    best_accuracy = 0.0
    best_model_path = 'best_model.pth'
    with torch.no_grad():
        for step, (X1, X2, y) in enumerate(test_iter):
            net.eval()
            X1 = X1.to(device)
            X2 = X2.to(device)
            y = y.to(device)
            y_hat = net(X1, X2)
            y_pred.extend(y_hat.cpu().argmax(dim=1))
            y_test.extend(y.cpu())
            net.train()
    if dataset == 'Houston':
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Tree',
                    'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                    'Railway', 'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track']
    if dataset == 'Berlin':
        target_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil',
                        'Allotment', 'Commercial Area', 'Water']
    if dataset == 'Trento':
        target_names = ['Apple trees', 'Buildings', 'Ground', 'Wood', 'Vineyard', 'Roads']
    if dataset == 'MUUFL':
        target_names = ['Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand', 'Road', 'Water',
                        'Building shadow', 'Building', 'Sidewalk', 'Yellow curb', 'Cloth panels']
    classification = classification_report(np.array(y_test), np.array(y_pred), target_names=target_names, digits=4)
    oa = accuracy_score(np.array(y_test), np.array(y_pred))
    confusion = confusion_matrix(np.array(y_test), np.array(y_pred))
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.array(y_test), np.array(y_pred))
    if oa > best_accuracy:
        best_accuracy = oa
        torch.save(net.state_dict(), best_model_path)
        print(f'Model saved with accuracy:{oa: .2f}%')

    print(oa, aa, kappa, each_acc)
    return oa, aa, kappa, each_acc


def pad_with_zeros(X, margin=2):
    """apply zero padding to X with margin(w, h, c)"""  #(c, w, h)

    new_X = np.zeros((X.shape[0], X.shape[1] + 2 * margin, X.shape[2] + 2 * margin))
    x_offset = margin
    y_offset = margin
    new_X[:, x_offset:X.shape[1] + x_offset, y_offset:X.shape[2] + y_offset] = X
    return new_X

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

def generate_image_cubes_all(input_data, pixels_select, windowSize=11): # 
    height, width, Band = input_data.shape
    kind = np.unique(pixels_select).shape[0]

    padding_data = np.pad(input_data, ((windowSize // 2, windowSize // 2),
                                       (windowSize // 2, windowSize // 2),
                                       (0, 0)), "constant")
    padding_label = np.pad(pixels_select, ((windowSize // 2, windowSize // 2),
                                           (windowSize // 2, windowSize // 2)), "constant")

    for i in range(height):
        for j in range(width):
            row_start = i
            row_end = i + windowSize
            col_start = j
            col_end = j + windowSize

            cube = padding_data[row_start:row_end, col_start:col_end, :]
            cube = cube.swapaxes(0, 2)

            label = np.zeros([kind])
            label_idx = padding_label[i + windowSize // 2, j + windowSize // 2] - 1
            label[label_idx] = 1
            label = np.argmax(label)

            yield cube, label
def generate_image_cubes(input_data, pixels_select, windowSize=11):
    Band = input_data.shape[2]
    kind = np.unique(pixels_select).shape[0] - 1

    padding_data = np.pad(input_data, ((windowSize // 2, windowSize // 2),
                                       (windowSize // 2, windowSize // 2),
                                       (0, 0)), "constant")
    padding_label = np.pad(pixels_select, ((windowSize // 2, windowSize // 2),
                                           (windowSize // 2, windowSize // 2)), "constant")
    pixel = np.where(padding_label != 0)

    for i in range(len(pixel[0])):
        row_start = pixel[0][i] - windowSize // 2
        row_end = pixel[0][i] + windowSize // 2 + 1
        col_start = pixel[1][i] - windowSize // 2
        col_end = pixel[1][i] + windowSize // 2 + 1

        cube = padding_data[row_start:row_end, col_start:col_end, :]
        cube = rearrange(cube, 'h w c -> c h w')

        label = np.zeros([kind])
        label_idx = padding_label[pixel[0][i], pixel[1][i]] - 1
        label[label_idx] = 1
        label = np.argmax(label)

        yield cube, label


def get_image_cubes(input_data, pixels_select, windowSize=11):
    gen = generate_image_cubes(input_data, pixels_select, windowSize)

    batch_out = []
    batch_label = []

    for cube, label in gen:
        batch_out.append(cube)
        batch_label.append(label)

    return np.array(batch_out), np.array(batch_label)

def get_image_cubes_all(input_data, pixels_select, windowSize=11):
    gen = generate_image_cubes_all(input_data, pixels_select, windowSize)

    batch_out = []
    batch_label = []

    for cube, label in gen:
        batch_out.append(cube)
        batch_label.append(label)

    return np.array(batch_out), np.array(batch_label)


def GetImageCubes_all(input_data, pixels_select, windowSize=11):  # 这里的label_select就是train_pixels/test_pixels
    height = input_data.shape[0]
    width = input_data.shape[1]
    Band = input_data.shape[2]
    kind = np.unique(pixels_select).shape[0]   # 得到测试或者训练集中的种类数
    # print(kind)
    # print('input_data.shape:', input_data.shape)
    paddingdata = np.pad(input_data, ((30, 30), (30, 30), (0, 0)), "constant")  # 采用边缘值填充 [205, 205, 200]                 可以作为超参数
    paddinglabel = np.pad(pixels_select, ((30, 30), (30, 30)), "constant")  # 此处"constant"应改为"edge"
    # 得到 label的 pixel坐标位置,去除背景元素
    y0 = np.ones((height, width))
    paddingx = np.pad(y0, ((30, 30), (30, 30)), "constant")  # 此处"constant"应改为"edge"
    pixel = np.where(paddingx != 0)
    # the number of batch
    num = np.sum(y0 != 0)  # 参与分类的像素点个数
    batch_out = np.zeros([num, windowSize, windowSize, Band])
    batch_label = np.zeros([num, kind])
    for i in range(num):  # 得到每个像素点的batch，在这里为19*19的方块
        row_start = pixel[0][i] - windowSize // 2
        row_end = pixel[0][i] + windowSize // 2 + 1
        col_start = pixel[1][i] - windowSize // 2
        col_end = pixel[1][i] + windowSize // 2 + 1
        batch_out[i, :, :, :] = paddingdata[row_start:row_end, col_start:col_end, :]  # 得到一个数据块
        temp = (paddinglabel[pixel[0][i], pixel[1][i]] - 1)  # temp = (label_selct[pixel[0][i],pixel[1][i]]-1)
        batch_label[i, temp] = 1  # 独热编码，并且是从零开始的
    # 修改合适三维卷积输入维度 [depth height weight]
    batch_out = batch_out.swapaxes(1, 3)
    batch_label = np.argmax(batch_label, axis=-1)
    # batch_out = batch_out[:, :, :, :, np.newaxis]           # np.newaxis:增加维度
    # print('batch_out.shape:', batch_out.shape)
    return batch_out, batch_label

def GetImageCubes(input_data, pixels_select, windowSize=11):  # 这里的label_select就是train_pixels/test_pixels
    Band = input_data.shape[2]
    kind = np.unique(pixels_select).shape[0] - 1  # 得到测试或者训练集中的种类数
    paddingdata = np.pad(input_data, ((30, 30), (30, 30), (0, 0)), "constant")  # 采用边缘值填充 [203, 203, 200]
    paddinglabel = np.pad(pixels_select, ((30, 30), (30, 30)), "constant")  # 此处"constant"应改为"edge"
    pixel = np.where(paddinglabel != 0)  # pixel = np.where(label_select != 0)  ，这里的pixel是坐标数据，不是光谱数据
    num = np.sum(pixels_select != 0)  # 参与分类的像素点个数
    batch_out = np.zeros([num, windowSize, windowSize, Band])
    batch_label = np.zeros([num, kind])
    for i in range(num):  # 得到每个像素点的batch，在这里为19*19的方块
        row_start = pixel[0][i] - windowSize // 2
        row_end = pixel[0][i] + windowSize // 2 + 1
        col_start = pixel[1][i] - windowSize // 2
        col_end = pixel[1][i] + windowSize // 2 + 1
        batch_out[i, :, :, :] = paddingdata[row_start:row_end, col_start:col_end, :]  # 得到一个数据块
        temp = (paddinglabel[pixel[0][i], pixel[1][i]] - 1)  # temp = (label_selct[pixel[0][i],pixel[1][i]]-1)
        batch_label[i, temp] = 1  # 独热编码，并且是从零开始的
    # 修改合适三维卷积输入维度 [depth height weight]
    batch_out = rearrange(batch_out, 'b h w c -> b c h w')
    #batch_out = batch_out.swapaxes(1, 3)
    batch_label = np.argmax(batch_label, axis=-1)
    # batch_out = batch_out[:, :, :, :, np.newaxis]           # np.newaxis:增加维度
    # print('batch_out.shape:', batch_out.shape)
    return batch_out, batch_label

def train_2_patch_label2(Data1, Data2, patchsize, pad_width, Label):
    [m1, n1, l1] = np.shape(Data1)
    [m2, n2, l2] = np.shape(Data2)
    # [m3, n3, l3] = np.shape(Data3)
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1  # 349*1905*144
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2  # 349*1905*21
    # for i in range(l3):
    #     Data3[:, :, i] = (Data3[:, :, i] - Data3[:, :, i].min()) / (Data3[:, :, i].max() - Data3[:, :, i].min())
    # x3 = Data3  # 349*1905*21
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')  # 365*1921*144
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')  # 365*1921*21
    # x3_pad = np.empty((m3 + patchsize, n3 + patchsize, l3), dtype='float32')  # 365*1921*21
    for i in range(l1):
        temp = x1[:, :, i]  # 349*1905
        temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
        x1_pad[:, :, i] = temp2  # 365*1921*144
    for i in range(l2):
        temp = x2[:, :, i]  # 349*1905
        temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
        x2_pad[:, :, i] = temp2  # 365*1921*21
    # for i in range(l3):
    #     temp = x3[:, :, i]  # 349*1905
    #     temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
    #     x3_pad[:, :, i] = temp2  # 365*1921*21
    # construct the training and testing set
    [ind1, ind2] = np.where(Label > 0)  # [300,300]  !=0  change > 0 ,muufl = -1
    TrainNum = len(ind1)  # 300
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')  # 300*144**16*16
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')  # 300*21**16*16
    # TrainPatch3 = np.empty((TrainNum, l3, patchsize, patchsize), dtype='float32')  # 300*144**16*16
    TrainLabel = np.empty(TrainNum)  # 300
    ind3 = ind1 + pad_width  # 300
    ind4 = ind2 + pad_width  # 300
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]  # 16*16*144
        patch1 = np.transpose(patch1, (2, 0, 1))  # 144*16*16
        TrainPatch1[i, :, :, :] = patch1  # 300*144*16*16
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]  # 16*16*21
        patch2 = np.transpose(patch2, (2, 0, 1))  # 21*16*16
        TrainPatch2[i, :, :, :] = patch2  # 300*21*16*16
        # patch3 = x3_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width),
        #          :]  # 16*16*21
        # patch3 = np.transpose(patch3, (2, 0, 1))  # 21*16*16
        # TrainPatch3[i, :, :, :] = patch3  # 300*21*16*16
        patchlabel = Label[ind1[i], ind2[i]]  # 1
        TrainLabel[i] = patchlabel  # 300
    # step3: change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    # TrainPatch3 = torch.from_numpy(TrainPatch3)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainLabel



def split_train_test_set(X, y, train_ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def weight_init(layer):
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)  # 获取confusion_matrix的主对角线所有数值
    list_raw_sum = np.sum(confusion_matrix, axis=1)  # 将主对角线所有数求和
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # list_diag/list_raw_sum  对角线各个数字/对角线所有数字的总和
    average_acc = np.mean(each_acc)  #
    return np.round(each_acc, 4), average_acc

def colormap(num_class, p):
    if p == True:
        # cdict = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#C0C0C0',
        #          '#808080', '#800000', '#808000', '#008000', '#800080', '#008080', '#000080', '#FFA500', '#FFD700']
        cdict = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']
        return colors.ListedColormap(cdict, N=num_class)
    else:
        # cdict = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#C0C0C0', '#808080',
        #         '#800000', '#808000', '#008000', '#800080', '#008080', '#000080', '#FFA500', '#FFD700']
        cdict = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#C86400', '#00C864',
                 '#6400C8', '#C80064', '#64C800', '#0064C8', '#964B4B', '#4B964B', '#4B4B96', '#FF6464']
        return colors.ListedColormap(cdict, N=num_class+1)

def dis_groundtruth(dataset, num_class, gt, p):
    '''plt.figure(title)
    plt.title(title)'''
    plt.imshow(gt, cmap=colormap(num_class, p))
    # spectral.imshow(classes=gt)
    '''plt.colorbar()'''
    plt.xticks([])
    plt.yticks([])
    '''plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)'''
    if p:
        plt.savefig('./results/{}/{}.png'.format(dataset, dataset+'true'), dpi=1200, pad_inches=0.0)
    else:
        plt.savefig('./results/{}/{}.png'.format(dataset, dataset+'false'), dpi=1200, pad_inches=0.0)
    plt.show()

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

def class_acc(y_true, y_pre):
    """
    calculate each class's acc
    :param y_true:
    :param y_pre:
    :return:
    """
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
        y_c_p = y_pre[np.nonzero(y_true == c)]
        acurracy = metrics.accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    return ca

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def cluster_accuracy(y_true, y_pre, return_aligned=False):
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]

    # # calculate accuracy
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
    kappa = metrics.cohen_kappa_score(y_true, y_best)
    ca = class_acc(y_true, y_best)
    ari = metrics.adjusted_rand_score(y_true, y_best)
    fscore = metrics.f1_score(y_true, y_best, average='micro')
    pur = purity_score(y_true, y_best)
    if return_aligned:
        return y_best, acc, kappa, nmi, ari, pur, ca
    return acc, kappa, nmi, ari, pur, ca