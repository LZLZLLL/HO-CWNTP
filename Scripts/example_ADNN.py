import os
import math
import torch
import numpy as np
import torch.nn as nn
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(size):
    path = 'D:/Experiment/meta_learning/data/' + str(cell) + '/' + str(cell) + '.txt'
    alldata = pd.read_table(path)
    data1 = alldata.fillna(0, inplace=False)
    data2 = data1.loc[data1['Internet'] > 0]
    df = data2.loc[data2['Country'] == 39]
    df3 = df.drop(columns=['Cell', 'Country', 'ReceivedSMS', 'SendSMS', 'IncomingCall', 'OutgoingCall'])
    df4 = df3.set_index(['Time'], inplace=False)
    df4 = df4.sort_values('Time')
    df5 = df4.reset_index()
    a_train = df5.shape[0]
    data18 = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5',
                                   '6', '7', '8', '9', '10', '11',
                                   '12', '13', '14', '15', '16', '17',
                                   'predict'])
    data6 = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5',
                                  'predict'])
    data12 = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5',
                                   '6', '7', '8', '9', '10', '11', 'predict'])
    for i in range(a_train - 18):
        d = df5['Internet'][i:i + 19]
        data18.loc[i] = [d[i], d[i + 1], d[i + 2], d[i + 3],
                         d[i + 4], d[i + 5], d[i + 6],
                         d[i + 7], d[i + 8], d[i + 9],
                         d[i + 10], d[i + 11], d[i + 12],
                         d[i + 13], d[i + 14], d[i + 15],
                         d[i + 16], d[i + 17], d[i + 18]]

        data6.loc[i] = [d[i + 12], d[i + 13], d[i + 14], d[i + 15],
                        d[i + 16], d[i + 17], d[i + 18]]

        data12.loc[i] = [d[i + 6], d[i + 7], d[i + 8], d[i + 9],
                         d[i + 10], d[i + 11], d[i + 12],
                         d[i + 13], d[i + 14], d[i + 15],
                         d[i + 16], d[i + 17], d[i + 18]]
    if cc == 18:
        data = data18
    elif cc == 12:
        data = data12
    else:
        data = data6

    if size < 1:
        train_data, uesless = train_test_split(data, train_size=size, random_state=42)
    else:
        train_data = data.sample(frac=1, replace=True, random_state=42)

    path2 = 'D:/Experiment/meta_learning/data/' + str(cell) + '/' + str(cell) + 'test.txt'
    alldata_test = pd.read_table(path2)
    data1_test = alldata_test.fillna(0, inplace=False)
    data2_test = data1_test.loc[data1_test['Internet'] > 0]
    df_test = data2_test.loc[data2_test['Country'] == 39]
    df3_test = df_test.drop(columns=['Cell', 'Country', 'ReceivedSMS', 'SendSMS', 'IncomingCall', 'OutgoingCall'])
    df4_test = df3_test.set_index(['Time'], inplace=False)
    df4_test = df4_test.sort_values('Time')
    df5_test = df4_test.reset_index()
    a_test = df5_test.shape[0]
    data_test18 = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5',
                                        '6', '7', '8', '9', '10', '11',
                                        '12', '13', '14', '15', '16', '17',
                                        'predict'])
    data_test6 = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5',
                                       'predict'])
    data_test12 = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5',
                                        '6', '7', '8', '9', '10', '11', 'predict'])
    for i in range(a_test - 18):
        v = df5_test['Internet'][i:i + 19]
        data_test18.loc[i] = [v[i], v[i + 1], v[i + 2], v[i + 3],
                              v[i + 4], v[i + 5], v[i + 6],
                              v[i + 7], v[i + 8], v[i + 9],
                              v[i + 10], v[i + 11], v[i + 12],
                              v[i + 13], v[i + 14], v[i + 15],
                              v[i + 16], v[i + 17], v[i + 18]]

        data_test6.loc[i] = [v[i + 12], v[i + 13], v[i + 14], v[i + 15],
                             v[i + 16], v[i + 17], v[i + 18]]

        data_test12.loc[i] = [v[i + 6], v[i + 7], v[i + 8], v[i + 9],
                              v[i + 10], v[i + 11], v[i + 12],
                              v[i + 13], v[i + 14], v[i + 15],
                              v[i + 16], v[i + 17], v[i + 18]]
    if cc == 18:
        test_data = data_test18
    elif cc == 12:
        test_data = data_test12
    else:
        test_data = data_test6
    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    max1 = max(max(data_test18.max()), max(train_data.max()))
    min1 = min(min(data_test18.max()), min(train_data.max()))
    train_data = (train_data-min1) / (max1-min1)
    test_data = (test_data-min1) / (max1-min1)
    train_seq = torch.from_numpy(np.array(train_data.iloc[:, :-output_window]))
    train_label = torch.from_numpy(np.array(train_data.iloc[:, output_window:]))
    dec_input = train_seq
    test_seq = torch.from_numpy(np.array(test_data.iloc[:, :-output_window]))
    test_label = torch.from_numpy(np.array(test_data.iloc[:, output_window:]))
    test_dec_input = test_seq
    train_sequence = torch.stack((train_seq, dec_input, train_label,), dim=1).type(torch.FloatTensor)
    test_data = torch.stack((test_seq, test_dec_input, test_label), dim=1).type(torch.FloatTensor)

    return train_sequence.to(device), test_data.to(device), max1, min1


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()

    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)

        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)

        return nn.LayerNorm(d_model).to(device)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs.squeeze(-1), enc_inputs.squeeze(-1))
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs.squeeze(-1), dec_inputs.squeeze(-1)).to(device)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs.squeeze(-1)).to(device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs.squeeze(-1), enc_inputs.squeeze(-1))
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size).to(device)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)

        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns


def train(train_data):
    model.train()
    total_loss = 0.
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        enc_input, dec_input, target = get_batch(train_data, i, batch_size)
        output, _, _, _ = model(enc_input, dec_input)
        output = output.transpose(0, 1)
        target = target.transpose(0, 1)
        loss = criterion(output[-output_window:], target[-output_window:])
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = batch_size

    for i in range(0, len(data_source) - 1, eval_batch_size):
        enc_input, dec_input, target = get_batch(data_source, i, batch_size)
        output, _, _, _ = model(enc_input, dec_input)
        output = output.transpose(0, 1)
        target = target.transpose(0, 1)
        loss = criterion(output[-output_window:], target[-output_window:]).item()
        total_loss += loss

    return total_loss / len(data_source)


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    truth = torch.Tensor(0)
    total_result = torch.Tensor(0)
    total_loss = 0.

    with torch.no_grad():
        for i in range(0, len(data_source) - 1, batch_size):
            enc_input, dec_input, target = get_batch(data_source, i, batch_size)
            output, _, _, _ = model(enc_input, dec_input)
            output = output.transpose(0, 1)
            target = target.transpose(0, 1)
            total_result = torch.cat((total_result, output[-output_window:].squeeze(2).view(-1).cpu() * (max1-min1)+min1), 0)
            truth = torch.cat((truth, target[-output_window:].squeeze(2).view(-1).cpu() * (max1-min1)+min1), 0)
            loss = criterion(output[-output_window:], target[-output_window:])
            total_loss += loss.item()

    return total_loss / len(data_source), total_result, truth


cc = 18
lr = 0.001
n_layers = 2
n_heads = 8
d_model = 512
d_ff = 2048
tgt_vocab_size = 1
input_window = 31
output_window = 1
batch_size = 43
d_k = d_v = 8
size = 0.8
epochs = 600
times = 10
cells = [1111]
for cell in cells:
    outputpath = 'D:/Experiment/meta_learning/cell' + str(k1)
    os.makedirs(outputpath + '/cell' + str(cell), exist_ok=True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    training_data, val_data, max1, min1 = get_data(size)
    pred = []
    tru_value = []
    for e in range(times):

        model = Transformer().to(device)
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.96)
        b = []
        predict = torch.Tensor(0)
        groundtruth = torch.Tensor(0)
        for epoch in range(epochs):
            train(training_data)
            train_loss = evaluate(model, training_data)
            test_loss, predict, groundtruth = plot_and_loss(model, val_data, epoch)
            b.append(test_loss)
            scheduler.step()
            if (epoch + 1) > 100 and test_loss < 0.01 and torch.tensor(b[-30:]).max() - torch.tensor(
                    b[-30:]).min() < 0.00005:
                print('convergence')
                break

        pred.append(predict)
        np.savetxt(outputpath + '/cell' + str(cell) + '/predict_value' + str(e) + '.csv', pred[e].numpy())
        tru_value.append(groundtruth)

        predict = predict.view(-1).data.numpy()
        groundtruth = groundtruth.view(-1).data.numpy()

        MSE = mean_squared_error(groundtruth, predict)
        print('MSE:', MSE)

        RMSE = np.sqrt(mean_squared_error(groundtruth, predict))
        print('RMSE:', RMSE)

        MAE = mean_absolute_error(groundtruth, predict)
        print('MAE:', MAE)

        r2 = r2_score(groundtruth, predict)
        print("r2 coefficient:", r2)

    np.savetxt(outputpath + '/cell' + str(cell) + '/groundtruth.csv', tru_value[0].numpy())
