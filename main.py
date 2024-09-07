# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2022年09月22日
"""
import os
import torch
from dataset import load_data, generater, normalize, setup_seed
from module import weight_init, AA_andEachClassAccuracy, applyPCA
from net import Network
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
import time

# 配置可调变量
config = {
    'dataset': 'Houston',  # 'Houston', 'Trento', 'MUUFL'
    'channel': 144,
    'class_num': 15,
    'batch_size': 128,
    'window_size': 11,
    'learning_rate': 0.0005,
    'weight_decay': 0.0001,
    'epoches': 300,
    'lr_step_size': 30,
    'lr_gamma': 0.5,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'seed': 100
}

def train_model(net, epoches, train_iter, optimizer, criterion, device, dataset):
    train_loss_list = []
    train_acc_list = []
    best_accuracy = 0.0
    best_y_pred = []
    best_y_test = []
    best_model_path = f'./models/{dataset}.pt'
    tick1 = time.time()

    for epoch in range(epoches):
        train_acc_sum, train_loss_sum = 0.0, 0.0
        for step, (x1, x2, y) in enumerate(train_iter):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_hat = net(x1, x2)
            loss = criterion(y_hat, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=-1) == y).float().sum().item()

        lr_adjust.step()

        print(
            f'epoch {epoch + 1}, train loss {train_loss_sum / len(train_iter):.6f}, train acc {train_acc_sum / len(train_iter.dataset):.4f}')
        train_loss_list.append(train_loss_sum / len(train_iter))
        train_acc_list.append(train_acc_sum / len(train_iter.dataset))

        if train_loss_list[-1] <= min(train_loss_list):
            print('\n***Start Testing***\n')
            y_test = []
            y_pred = []
            net.eval()
            with torch.no_grad():
                for step, (x1, x2, y) in enumerate(test_iter):
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    y_hat = net(x1, x2)
                    y_pred.extend(y_hat.cpu().argmax(dim=1))
                    y_test.extend(y.cpu())

            net.train()
            oa = accuracy_score(np.array(y_test), np.array(y_pred))
            confusion = confusion_matrix(np.array(best_y_test), np.array(best_y_pred))
            each_acc, aa = AA_andEachClassAccuracy(confusion)
            kappa = cohen_kappa_score(np.array(best_y_test), np.array(best_y_pred))

            if oa > best_accuracy:
                best_accuracy = oa
                best_y_pred = y_pred.copy()
                best_y_test = y_test.copy()
                print('***Saving model parameters***')
                torch.save(net.state_dict(), best_model_path)

            print(f'OA: {oa}, AA: {aa}, Kappa: {kappa}, Best accuracy: {best_accuracy}')

    tick2 = time.time()
    save_classification_report(dataset, best_y_test, best_y_pred, best_accuracy, aa, kappa, tick2 - tick1)


def save_classification_report(dataset, best_y_test, best_y_pred, best_accuracy, aa, kappa, training_time):
    target_names = get_target_names(dataset)
    classification = classification_report(np.array(best_y_test), np.array(best_y_pred), target_names=target_names,
                                           digits=4)
    confusion = confusion_matrix(np.array(best_y_test), np.array(best_y_pred))

    file_name = f"./results/{dataset}/{dataset}.txt"
    with open(file_name, 'a') as x_file:
        x_file.write('\n' + '*' * 90 + '\n')
        x_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        x_file.write(f'Overall accuracy (OA): {best_accuracy * 100:.2f}%\n')
        x_file.write(f'Average accuracy (AA): {aa * 100:.2f}%\n')
        x_file.write(f'Kappa accuracy: {kappa * 100:.2f}%\n\n')
        x_file.write(f'{classification}\n')
        x_file.write(f'{confusion}\n')
        x_file.write(f'Training Time: {training_time}s\n')
        x_file.write('\n' + '*' * 90 + '\n')


def get_target_names(dataset):
    if dataset == 'Houston':
        return ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Tree', 'Soil', 'Water',
                'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking lot 1',
                'Parking lot 2', 'Tennis court', 'Running track']
    elif dataset == 'Berlin':
        return ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil', 'Allotment',
                'Commercial Area', 'Water']
    elif dataset == 'Trento':
        return ['Apple trees', 'Buildings', 'Ground', 'Wood', 'Vineyard', 'Roads']
    elif dataset == 'MUUFL':
        return ['Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand', 'Road', 'Water',
                'Building shadow', 'Building', 'Sidewalk', 'Yellow curb', 'Cloth panels']


# 设置随机种子
setup_seed(config['seed'])

# 加载数据并预处理
HSI_data, LiDAR_data, Train_data, Test_data, GT = load_data(config['dataset'])
HSI_data = normalize(HSI_data, type=1)
LiDAR_data = normalize(LiDAR_data, type=1)
(TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter) = generater(
    HSI_data, LiDAR_data, Train_data, Test_data, GT,
    batch_size=config['batch_size'], windowSize=config['window_size']
)

# 打印数据集信息
print(f'TRAIN_SIZE: {TRAIN_SIZE}')
print(f'TEST_SIZE: {TEST_SIZE}')
print(f'TOTAL_SIZE: {TOTAL_SIZE}')
print(f'----Training on {config["device"]}----\n')

# 初始化网络和训练配置
net = Network(config['channel'], config['window_size'], config['class_num'], config['dataset']).to(config['device'])
net.apply(weight_init)
optimizer = optim.AdamW(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
criterion = nn.CrossEntropyLoss().to(config['device'])
lr_adjust = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config['lr_step_size'],
                                            gamma=config['lr_gamma'])

# 创建模型保存文件夹
if not os.path.exists('./models'):
    os.makedirs('./models')

# 训练模型
train_model(net, config['epoches'], train_iter, optimizer, criterion, config['device'], config['dataset'])
