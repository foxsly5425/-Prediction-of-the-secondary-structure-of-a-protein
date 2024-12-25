class ProteinLSTM(nn.Module):
    def __init__(self, embedding_layer, output_dim):
        super(ProteinLSTM, self).__init__()
        self.embedding = embedding_layer

        input_dim      = embedding_layer.embedding_dim


        self.lstm = nn.LSTM(256, 256, num_layers=2,
                            bidirectional=True, batch_first=True)
        
        self.fc   = nn.Linear(256 * 2, 256 * 2)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(256 * 2, output_dim)

    def forward(self, x):
        x    = self.embedding(x)
        x, _ = self.lstm(x)       
        x    = self.fc(x)
        x    = self.relu(x)
        x    = self.fc2(x)
        return x
