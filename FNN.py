class ProteinFNN(nn.Module):
    def __init__(self, embedding_layer, output_dim, max_len=2000):
        super(ProteinFNN, self).__init__()
        self.embedding = embedding_layer

        input_dim      = embedding_layer.embedding_dim

        self.layer1    = nn.Linear(input_dim, 1024)
        self.layer2    = nn.Linear(1024, 1024)
        self.layer3    = nn.Linear(1024, output_dim)
        self.relu      = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.embedding(x)

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)

        return x
