#coding = UTF-8
import sys
import scipy.io as scio
import os
import argparse
import numpy as np
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # 导入包
import matplotlib.font_manager as fm
from torch.utils.data import Dataset
from get_data import MyDataset
from neural_network_models.cnn_model import CNN
from neural_network_models.mlp_model import MLP
from neural_network_models.rnn_model import RNN
from neural_network_models.cnn_lstm_model import CNN_LSTM

def test(model, dataset, criterion):
    model.eval()
    total_batch_num = 0
    val_loss = 0
    prediction = []
    labels = []
    feature_list = torch.tensor([])   
    for (step, i) in enumerate(dataset):
        total_batch_num = total_batch_num+1
        batch_x = i['data']
        batch_y = i['label']
        batch_x = torch.unsqueeze(batch_x, dim=1)  # (50, 1, 10000, 12)
        batch_x = batch_x.float()
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        feature, probs = model(batch_x)# feature  20,400
        batch_label = batch_y.unsqueeze(1).float()
        feature_label = torch.cat((feature, batch_label), dim=1)
        feature_list = torch.cat((feature_list, feature_label), dim=0)   # feature_list
        loss = criterion(probs, batch_y)
        _, pred = torch.max(probs, dim=1)
        predi = pred.tolist()
        label = batch_y.tolist()
        val_loss += loss.item()
        prediction.extend(predi)
        labels.extend(label)
    accuracy = accuracy_score(labels, prediction)
    C = confusion_matrix(labels, prediction)
    return accuracy, val_loss/total_batch_num, feature_list, C


def train(model, train_x, train_y, optimizer, criterion):
    model.train()
    model.zero_grad()

    if isinstance(model, RNN):
        # Reshape the input for RNN
        batch_size, channels, height, width = train_x.size()
        train_x = train_x.view(batch_size, height, width)  # Reshape to [batch_size, seq_len, input_size]

    _, probs = model(train_x)
    loss = criterion(probs, train_y)
    _, pred = torch.max(probs, dim=1)
    labels = train_y.tolist()
    predi = pred.tolist()
    loss.backward()
    optimizer.step()
    return labels, predi, loss.item()



class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger('result.log', sys.stdout)


def main(args):
    train_dataset = MyDataset(args.root, args.txtpath, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = MyDataset(args.root2, args.txtpath2, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    models = {"CNN": CNN, "MLP": MLP, "RNN": RNN, "CNN_LSTM": CNN_LSTM}
    # In train_test.py
    if args.model == "RNN":
        model = RNN(input_size=120, hidden_size=128, num_layers=2, num_classes=6)
    else:
        model = models[args.model]()

    model = models[args.model]()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    batches_per_epoch = int(len(train_dataset)/args.batch_size)
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    train_time = 0
    for epoch in range(args.epochs):
        tic = time.time()
        train_predict = []
        train_label = []
        runloss = 0
        for (cnt, i) in enumerate(train_loader):
            batch_x = i['data']
            batch_y = i['label']
            batch_x = torch.unsqueeze(batch_x, dim=1)
            batch_x = batch_x.float()
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            tlabels, tpredi, tloss = train(model, batch_x, batch_y, optimizer, criterion)
            runloss = runloss+tloss
            train_label.extend(tlabels)
            train_predict.extend(tpredi)
        per_epoch_train_time = time.time()-tic
        train_time = per_epoch_train_time+train_time
        taccuracy = accuracy_score(train_label, train_predict)
        train_acc_list.append(taccuracy)
        loss = runloss / batches_per_epoch
        train_loss_list.append(loss)
        acc_score, loss_score, feature_list, C = test(model, test_loader, criterion)
        print("Epoch %d Val_accuracy %.3f Val_loss %.3f" % (epoch, acc_score, loss_score))
        test_acc_list.append(acc_score)
        test_loss_list.append(loss_score)
        if epoch == args.epochs-1:  
            torch.save(model, 'model.pth')
            the_model = torch.load('model.pth', weights_only=False)
            acc_score, loss_score, NAR, feature_list = test(the_model, test_loader, criterion)
            print("Epoch %d Train_accuracy %.3f Train_loss %.3f Val_accuracy %.3f Val_loss %.3f "              
                % (epoch, taccuracy, loss, acc_score, loss_score))
            if isinstance(feature_list, torch.Tensor):
                feature_list = feature_list.detach().numpy()
            np.savetxt('feature_data.csv', feature_list, delimiter=',')  
            draw_result(C)
            
            print('train totally using %.3f seconds ', train_time)
    

def draw_result(C):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    df = pd.DataFrame(C)
    sns.heatmap(df, fmt='g', annot=True, robust=True,
                annot_kws={'size': 10},
                xticklabels=['background', 'digging', 'knocking', 'watering', 'shaking', 'walking'],
                yticklabels=['background', 'digging', 'knocking', 'watering', 'shaking', 'walking'],
                cmap= 'Blues')
    ax.set_xlabel('Predicted label', fontsize=15)
    ax.set_ylabel('True label', fontsize=15) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    plt.savefig('./CNN_confusion_matrix.jpg')
    plt.show()


    Acc = (C[0][0] + C[1][1] + C[2][2] + C[3][3] + C[4][4] + C[5][5]) / sum(C[0] + C[1] + C[2] + C[3] + C[4] + C[5])
    print('acc: %.4f' % Acc)
    NAR = (sum(C[0]) - C[0][0]) / sum(C[:,1]+C[:,2]+C[:,3]+C[:,4]+C[:,5])
    print('NAR: %.4f' % NAR)
    FNR = (sum(C[:,0]) - C[0][0]) / sum(C[1]+C[2]+C[3]+C[4]+C[5])
    print('FNR: %.4f' % FNR)
    column_sum = np.sum(C,axis=0)
    print(column_sum)
    row_sum =np.sum(C,axis=1)
    print(row_sum)

    for i in range(1, 6):
        Precision = C[i - 1][i - 1] / column_sum[i - 1]
        Recall = C[i - 1][i - 1] / row_sum[i - 1]
        F1=2*Precision*Recall/(Precision+Recall)
        print('precision_%d: %.3f' % (i, Precision))
        print('Recall_%d: %.3f' % (i, Recall))
        print('F1_%d: %.3f' % (i, F1))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CNN fou classification")

    '''save model'''
    parser.add_argument("--save", type=str, default="__",
                        help="path to save model")
    '''model parameters'''
    rootpath = 'das_data'
    parser.add_argument("--root", type=str, default=rootpath + '/train',
                        help="rootpath of traindata")
    parser.add_argument("--root2", type=str, default=rootpath + '/test',
                        help="rootpath of valdata")
    parser.add_argument("--txtpath", type=str, default=rootpath + '/train/label.txt',
                        help="path of train_list")
    parser.add_argument("--txtpath2", type=str, default=rootpath + '/test/label.txt',
                        help="path pf val_list")
    parser.add_argument("--model", type=str, default="CNN",
                        help="type of model to use for classification")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=15,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="batch size")
    my_args = parser.parse_args()

    main(my_args)

