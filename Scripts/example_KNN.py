import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error  # MAE
import datetime
from sklearn.metrics import r2_score

startTime = datetime.datetime.now()
output_window = 1
input_window = 4
inputlen = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outputpath = 'D:/Experiment/meta_learning/data/meta_sample/KNN'

os.makedirs(outputpath, exist_ok=True)


# Data
def get_data():
    path_train = 'D:/Experiment/meta_learning/data/meta_sample/KNN/TrainingSet.csv'
    data1 = pd.read_csv(path_train)
    data1 = data1.fillna(0, inplace=False)
    train_data = data1.drop(columns=['cell'])

    path_test = 'D:/Experiment/meta_learning/data/meta_sample/KNN/TestingSet.csv'
    data2 = pd.read_csv(path_test)
    data2 = data2.fillna(0, inplace=False)
    test_data = data2.drop(columns=['cell'])

    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)

    train_seq = torch.from_numpy(np.array(train_data.iloc[:, :-output_window]))
    train_label = torch.from_numpy(np.array(train_data.iloc[:, output_window:]))
    dec_input = train_seq

    test_seq = torch.from_numpy(np.array(test_data.iloc[:, :-output_window]))
    test_label = torch.from_numpy(np.array(test_data.iloc[:, output_window:]))
    test_dec_input = test_seq

    train_sequence = torch.stack((train_seq, dec_input, train_label,), dim=1).type(torch.FloatTensor)
    test_data = torch.stack((test_seq, test_dec_input, test_label), dim=1).type(torch.FloatTensor)
    max1 = 1
    return train_sequence.to(device), test_data.to(device), max1


def get_batch(source, i, batch_size):
    if batch_size < len(source) - 1 - i:
        data = source[i:i + batch_size]
    else:
        data = source[i:]

    enc_input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    dec_input = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[2] for item in data]).chunk(input_window, 1))

    enc_input = enc_input.transpose(0, 1)
    dec_input = dec_input.transpose(0, 1)
    target = target.transpose(0, 1)

    return enc_input, dec_input, target


class PoswiseFeedForwardNet1(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet1, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(inputlen, 512, bias=True),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024, bias=True),
            nn.ReLU()
        )
        self.hidden22 = nn.Sequential(
            nn.Linear(1024, 2048, bias=True),
            nn.ReLU()
        )
        self.hidden222 = nn.Sequential(
            nn.Linear(2048, 64, bias=True),
            nn.ReLU()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(64, 1, bias=True),
            nn.ReLU()
        )

    def forward(self, inputs):
        fc1 = self.hidden1(inputs)
        fc2 = self.hidden2(fc1)
        fc3 = self.hidden22(fc2)
        fc4 = self.hidden222(fc3)
        output = self.hidden3(fc4)
        return output


# 3层NN
class PoswiseFeedForwardNet2(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet2, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(inputlen, 512, bias=True),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 128, bias=True),
            nn.ReLU()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 1, bias=True),
            nn.ReLU()
        )

    def forward(self, inputs):
        fc1 = self.hidden1(inputs)
        fc2 = self.hidden2(fc1)
        output = self.hidden3(fc2)
        return output


model = PoswiseFeedForwardNet1().to(device)
criterion = nn.MSELoss(reduction='sum')
lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.96)


def train(train_data):
    model.train()
    total_loss = 0.

    start_time = time.time()
    count = 0
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        enc_input, dec_input, target = get_batch(train_data, i, batch_size)
        enc_input = enc_input.squeeze(-1)
        output = model(enc_input)
        output = output.transpose(0, 1)
        target = target.transpose(0, 1)
        target = target.squeeze(-1)
        loss = criterion(output[-output_window:], target[-output_window:])
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += len(enc_input)
        log_interval = int(len(train_data) / batch_size / 10)
        if (batch + 1) % log_interval == 0:
            cur_loss = total_loss / count
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'train loss {:5.5f}'.format(
                epoch, batch + 1, len(train_data) // batch_size, scheduler.get_last_lr()[0],
                       elapsed * 1000 / log_interval, cur_loss))
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = batch_size

    for i in range(0, len(data_source) - 1, eval_batch_size):
        enc_input, dec_input, target = get_batch(data_source, i, batch_size)
        enc_input = enc_input.squeeze(-1)
        output = model(enc_input)
        output = output.transpose(0, 1)
        target = target.transpose(0, 1)
        target = target.squeeze(-1)
        loss = criterion(output[-output_window:], target[-output_window:]).item()
        total_loss += loss

    return total_loss / len(data_source)


train_data, val_data, max1 = get_data()
batch_size = 10
epochs = 50
a = []
b = []
for epoch in range(epochs):
    epoch_start_time = time.time()
    train(train_data)
    train_loss = evaluate(model, train_data)

    time1 = datetime.datetime.now()
    truth = torch.Tensor(0)
    total_result = torch.Tensor(0)
    test_loss = 0.

    with torch.no_grad():
        for batch, i in enumerate(range(0, len(val_data) - 1, batch_size)):
            enc_input, test_dec_input, target = get_batch(val_data, i, batch_size)
            enc_input = enc_input.squeeze(-1)
            print(enc_input.shape)
            output = model(enc_input)
            output = output.transpose(0, 1)
            target = target.transpose(0, 1)
            total_result = torch.cat((total_result, output[-output_window:].squeeze(-1).view(-1).cpu() * max1), 0)
            print(total_result.shape)

            truth = torch.cat((truth, target[-output_window:].squeeze(-1).view(-1).cpu() * max1), 0)
            print(truth.shape)
            target = target.squeeze(-1)
            loss = criterion(output[-output_window:], target[-output_window:])
            test_loss += loss.item()
    test_loss = test_loss / len(val_data)
    a.append(train_loss)
    b.append(test_loss)
    scheduler.step()
    time2 = datetime.datetime.now()

    if (epoch + 1) % 5 == 0:
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.5f} | train loss {:5.5f} '.format(
            epoch, (time.time() - epoch_start_time), test_loss, train_loss))
        print('-' * 89)

torch.save(model, outputpath + '\epoch' + str(epochs) + '\\net.pkl')
pyplot.plot(a, color="blue", alpha=0.5)
pyplot.plot(b, color="red", alpha=0.5)
pyplot.grid(True, which='both')
pyplot.savefig(outputpath + '\epoch' + str(epochs) + '\Perf_Versus_Distance\Trainloss & testloss')
pyplot.close()
