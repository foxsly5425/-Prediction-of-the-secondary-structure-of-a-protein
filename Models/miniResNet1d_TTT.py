class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2   = nn.BatchNorm1d(out_channels)

        if in_channels   != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = self.relu(x)

        x  = self.conv2(x)
        x  = self.bn2(x)


        x += identity
        x  = self.relu(x)
        return x

class TTTLayerDualSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        super(TTTLayerDualSequence, self).__init__()
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.learning_rate = learning_rate

        self.theta_K = nn.Parameter(torch.eye(hidden_dim, input_dim))
        self.theta_V = nn.Parameter(torch.eye(hidden_dim, input_dim))
        self.theta_Q = nn.Parameter(torch.eye(hidden_dim, input_dim))

        self.W       = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.b       = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_batch):
        batch_size, seq_length, input_dim = x_batch.size()

        X       = x_batch.reshape(batch_size * seq_length, input_dim)

        X_train = F.linear(X, self.theta_K)
        X_label = F.linear(X, self.theta_V)
        X_test  = F.linear(X, self.theta_Q)

        WX         = torch.matmul(self.W, X.t())
        WX_minus_X = WX - X.t()

        grad_sum   = torch.matmul(WX_minus_X, X)

        Z = F.linear(X_test, self.W, self.b)

        Z = Z.view(batch_size, seq_length, self.hidden_dim)

        return Z

class BidirectionalTTTLayerDualSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        super(BidirectionalTTTLayerDualSequence, self).__init__()
        self.forward_ttt  = TTTLayerDualSequence(input_dim, hidden_dim, learning_rate)
        self.backward_ttt = TTTLayerDualSequence(input_dim, hidden_dim, learning_rate)

    def forward(self, x_batch):
        forward_out  = self.forward_ttt(x_batch)  

        backward_x   = torch.flip(x_batch, dims=[1])  
        backward_out = self.backward_ttt(backward_x)  
        backward_out = torch.flip(backward_out, dims=[1])  

        out = torch.cat((forward_out, backward_out), dim=2) 
        return out

class ProteinTTT(nn.Module):
    def __init__(self, embedding_layer, output_dim):
        super(ProteinTTT, self).__init__()
        self.embedding = embedding_layer

        input_dim      = embedding_layer.embedding_dim

        self.ttt       = BidirectionalTTTLayerDualSequence(input_dim=input_dim, hidden_dim=256, learning_rate=0.01)


        self.fc       = nn.Linear(256 * 2, 256 * 2)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Linear(256 * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.ttt(x)       
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ProteinResNetTTT(nn.Module):
    def __init__(self, embedding_layer, output_dim):
        super(ProteinResNetTTT, self).__init__()
        self.embedding  = embedding_layer

        input_dim       = embedding_layer.embedding_dim

        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256)
        self.ttt    = BidirectionalTTTLayerDualSequence(input_dim=256, hidden_dim=256, learning_rate=0.01)


        self.fc     = nn.Linear(256 * 2, output_dim)

    def forward(self, x):
        x   = self.embedding(x)  
        x   = x.permute(0, 2, 1) 

        x   = self.input_conv(x)
        x   = self.layer1(x)
        x   = self.layer2(x)
        x   = self.layer3(x)

        x   = x.permute(0, 2, 1)  
        ttt = self.ttt(x)       
        x   = self.fc(ttt_out)
        return out
