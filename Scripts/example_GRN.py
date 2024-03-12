import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data():
    path = 'D:/Experiment/meta_learning/data/GRN/NormTraining.csv'
    alldata = pd.read_csv(path)
    train_seq = torch.from_numpy(np.array(alldata.iloc[:, :feature_size + 1]))
    dec_input = torch.from_numpy(np.array(alldata.iloc[:, feature_size + 1:feature_size + 1 + hyperp_size]))
    train_label = torch.from_numpy(np.array(alldata.iloc[:, feature_size + 1 + 1:]))

    test_seq = torch.from_numpy(np.array(alldata.iloc[:, :feature_size + 1]))
    test_dec_input = torch.from_numpy(np.array(alldata.iloc[:, feature_size + 1:feature_size + 1 + hyperp_size]))
    test_label = torch.from_numpy(np.array(alldata.iloc[:, feature_size + 1 + 1:]))

    train_sequence = torch.stack((train_seq, dec_input, train_label,), dim=1).type(torch.FloatTensor)
    test_data = torch.stack((test_seq, test_dec_input, test_label), dim=1).type(torch.FloatTensor)

    return train_sequence.to(device), test_data.to(device)


def get_batch(source, i, batch_size):
    if batch_size < len(source) - 1 - i:
        data = source[i:i + batch_size]
    else:
        data = source[i:]
    enc_input = torch.stack([item[0] for item in data])[:, 1:]
    enc_input = enc_input.unsqueeze(1)
    dec_input = torch.stack([item[1] for item in data])
    dec_input = dec_input.unsqueeze(1)
    target = (torch.stack([item[2] for item in data]))[:, -1].reshape((-1, 1))
    return enc_input, dec_input, target


class GRN_metafeature(nn.Module):
    def __init__(self):
        super(GRN_metafeature, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_size, neure, bias=True),
            nn.ELU(),
            nn.Linear(neure, feature_size, bias=True)
        )
        self.gate1 = nn.Sequential(
            nn.Linear(feature_size, feature_size, bias=True),
            nn.Sigmoid()
        )
        self.gate2 = nn.Linear(feature_size, feature_size, bias=True)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = self.gate1(output) * self.gate2(output)
        return nn.LayerNorm(feature_size).to(device)(output + residual)


class GRN_hyperp(nn.Module):
    def __init__(self):
        super(GRN_hyperp, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hyperp_size, neure, bias=True),
            nn.ELU(),
            nn.Linear(neure, hyperp_size, bias=True)
        )
        self.gate1 = nn.Sequential(
            nn.Linear(hyperp_size, hyperp_size, bias=True),
            nn.Sigmoid()
        )
        self.gate2 = nn.Linear(hyperp_size, hyperp_size, bias=True)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = self.gate1(output) * self.gate2(output)
        return nn.LayerNorm(hyperp_size).to(device)(output + residual)


class GRN_fusion(nn.Module):
    def __init__(self):
        super(GRN_fusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_size + hyperp_size, neure, bias=True),
            nn.ELU(),
            nn.Linear(neure, feature_size + hyperp_size, bias=True)
        )
        self.gate1 = nn.Sequential(
            nn.Linear(feature_size + hyperp_size, feature_size + hyperp_size, bias=True),
            nn.Sigmoid()
        )
        self.gate2 = nn.Linear(feature_size + hyperp_size, feature_size + hyperp_size, bias=True)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = self.gate1(output) * self.gate2(output)
        return nn.LayerNorm(feature_size + hyperp_size).to(device)(output + residual)


class GRN_module(nn.Module):
    def __init__(self):
        super(GRN_module, self).__init__()
        self.mf_GRN = GRN_metafeature()
        self.mf_importance = GRN_metafeature()
        self.softmax_layer = nn.Softmax(dim=2)
        self.hp_GRN = GRN_hyperp()
        self.fusion = GRN_fusion()
        self.fc1 = nn.Sequential(
            nn.Linear(feature_size + hyperp_size, neure, bias=True).to(device),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(neure, neure, bias=True).to(device),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(neure, 1, bias=True).to(device)
        )

    def forward(self, metafeture, hyperparameter):
        transformed_mf = self.mf_GRN(metafeture)
        importance = self.softmax_layer(self.mf_importance(metafeture))
        trans_mf = torch.multiply(transformed_mf, importance)
        trans_hp = self.hp_GRN(hyperparameter)
        concatvector = torch.cat((trans_mf, trans_hp), 2)
        fusion = self.fusion(concatvector)
        midput1 = self.fc1(fusion)
        midput2 = self.fc2(midput1)
        output = self.fc3(midput2)

        return output.squeeze(1)


feature_size = 4
hyperp_size = 5
neure = 512
lr = 0.001
num_epochs = 600
batch_size = 252

train_data, val_data = get_data()
model = GRN_module().to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

trainmseloss = []
for epoch in range(num_epochs):
    trainloss = 0
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        enc_input, dec_input, target = get_batch(train_data, i, batch_size)
        outputs = model(enc_input, dec_input)

        optimizer.zero_grad()
        loss = criterion(outputs, target)
        trainloss += loss.item()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {trainloss}')
        trainmseloss.append(trainloss)
    if trainloss == min(trainmseloss):
        outputpath = 'D:/Experiment/meta_learning/data/GRN'
        torch.save(model, outputpath + '/epoch' + str(num_epochs) + 'net.pkl')


