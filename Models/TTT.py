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
