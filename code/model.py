import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, dropout_rate=0.0):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2,2), (2,1), (0,1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2,2), (2,1), (0,1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True)
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh, dropout_rate),
            BidirectionalLSTM(nh, nh, nclass, dropout_rate)
        )

    def forward(self, input):
        # input: (batch, channel, height, width)
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # (batch, channel, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, channel)
        output = self.rnn(conv)
        return output

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, dropout_rate=0.0):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.dropout = nn.Identity()  # 不使用dropout

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # (T*b, nOut)
        output = self.dropout(output)  # 不使用dropout
        output = output.view(T, b, -1)
        return output 