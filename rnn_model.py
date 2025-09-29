import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_classes, rnn_type="rnn"):
        super().__init__()
        self.rnn_type = rnn_type
        
        self.initial = nn.Linear(input_dim, hidden_dim)
        
        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        else:
            raise ValueError("type must be one of: 'rnn', 'gru', 'lstm'")
        
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, batch, mask):
        x = self.initial(batch)
        z, _ = self.rnn(x)
        
        predicted_batch = self.decoder(z)
        
        lengths = mask.sum(dim=1).long()
        last_idx = lengths - 1
        z_final = z[torch.arange(z.size(0)), last_idx, :]
        logits = self.classifier(z_final)

        return predicted_batch, logits
