import torch
import torch.nn as nn

# Bidirectional LSTM AutoEncoder model class
class BiLSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        '''
        encoder와 decoder의 종류, 레이어 수 등을 정의
        '''
        super(BiLSTMAutoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.encoder = sRNNCell()
        self.decoder = nn.LSTM(hidden_size * 2, hidden_size * 2, num_layers, batch_first=True, bidirectional=True)
        # self.decoder = sRNNCell()
        self.output_layer = nn.Linear(hidden_size * 4, input_size)

    def forward(self, x):
        '''
        __init__에서 정의한 초기화 형태를 기반으로 입출력 과정을 정의
        '''
        # 인코더
        _, (hidden, _) = self.encoder(x)
        # hidden의 형태: (num_layers * num_directions, batch, hidden_size)
        # 두 방향의 은닉 상태를 결합
        hidden = hidden.view(self.num_layers, 2, x.size(0), self.hidden_size)
        hidden = torch.cat((hidden[:,0,:,:], hidden[:,1,:,:]), dim=2)
        # 디코더 입력 준비
        decoder_input = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        # 디코더
        decoded, _ = self.decoder(decoder_input)
        # decoded, _ = self.decoder(decoder_input, x)
        # 출력 차원 조정
        decoded = decoded.reshape(decoded.size(0), decoded.size(1), -1)
        output = self.output_layer(decoded)
        return output