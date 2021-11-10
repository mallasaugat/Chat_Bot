import torch.nn as nn

class Net(nn.Module):
    def __init__(self, Layers):
        super().__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers,Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))
        self.relu = nn.ReLU()
    def forward(self, x):
        L = len(self.hidden)
        for l, linear_model in enumerate(self.hidden):
            if l < L-1:
                x = self.relu(linear_model(x))
            else:
                x = linear_model(x)
        return x
 

class Net_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
    
        self.rnn1=nn.RNN(input_size, hidden_size)
        self.rnn2=nn.RNN(hidden_size, hidden_size)
        self.rnn3=nn.RNN(hidden_size, hidden_size)
        self.rnn4=nn.RNN(hidden_size, hidden_size)
        self.rnn5=nn.RNN(hidden_size, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x, lengths=None):
        x = x.unsqueeze(0)

        y, x = self.rnn1(x)
        x = self.relu(x)

        y, x = self.rnn2(x)
        x = self.relu(x)

        y, x = self.rnn3(x)
        x = self.relu(x)

        y, x = self.rnn4(x)
        x = self.relu(x)

        y, x = self.rnn5(x)
        x = self.relu(x)

        x = x.squeeze(0)

        
        return x

class Net_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
    
        self.rnn1=nn.LSTM(input_size, hidden_size)
        self.rnn2=nn.LSTM(hidden_size, hidden_size)
        self.rnn3=nn.LSTM(hidden_size, hidden_size)
        self.rnn4=nn.LSTM(hidden_size, hidden_size)
        self.rnn5=nn.LSTM(hidden_size, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x, lengths=None):
        x = x.unsqueeze(0)

        y, x = self.rnn1(x)
        x = self.relu(x)

        y, x = self.rnn2(x)
        x = self.relu(x)

        y, x = self.rnn3(x)
        x = self.relu(x)

        y, x = self.rnn4(x)
        x = self.relu(x)

        y, x = self.rnn5(x)
        x = self.relu(x)

        x = x.squeeze(0)

        
        return x

class Net_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
    
        self.rnn1=nn.GRU(input_size, hidden_size)
        self.rnn2=nn.GRU(hidden_size, hidden_size)
        self.rnn3=nn.GRU(hidden_size, hidden_size)
        self.rnn4=nn.GRU(hidden_size, hidden_size)
        self.rnn5=nn.GRU(hidden_size, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x, lengths=None):
        x = x.unsqueeze(0)

        y, x = self.rnn1(x)
        x = self.relu(x)

        y, x = self.rnn2(x)
        x = self.relu(x)

        y, x = self.rnn3(x)
        x = self.relu(x)

        y, x = self.rnn4(x)
        x = self.relu(x)

        y, x = self.rnn5(x)
        x = self.relu(x)

        x = x.squeeze(0)

        
        return x
