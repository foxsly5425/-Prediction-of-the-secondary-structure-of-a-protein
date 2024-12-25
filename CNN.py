class ProteinCNN(nn.Module):
    def __init__(self, embedding_layer, output_dim, max_len=2000):
        super(ProteinCNN, self).__init__()
        self.embedding = embedding_layer

        input_dim = embedding_layer.embedding_dim

        self.conv1   = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1     = nn.LayerNorm(128)

        self.conv2   = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2     = nn.LayerNorm(128)

        self.conv3   = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3     = nn.LayerNorm(256)

        self.fc      = nn.Linear(256, output_dim)

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x
