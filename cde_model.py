import os
import sys
import torch
import torch.nn as nn
import torchcde
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utilities import LipSwish

class CDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_classes):
        super().__init__()
        self.sde_type = "stratonovich"
        self.noise_type = "scalar"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # vector_field
        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            LipSwish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.linear_out = nn.Linear(hidden_dim, input_dim * hidden_dim)

        # forward
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.__dict__['X'] = torchcde.CubicSpline(self.coeffs, self.times)
    
    def vector_field(self, t, y):
        if t.dim() == 0:
            t_scalar = t.item() if t.dim() == 0 else t
            t = torch.full_like(y[:, 0], fill_value=t_scalar).unsqueeze(-1)
        yy = self.linear_in(torch.cat((t, y), dim=-1))
        yy = self.f_net(yy)
        return self.linear_out(yy).view(yy.shape[0], self.hidden_dim, self.input_dim)
    
    def forward(self, coeffs, times, mask):
        self.set_X(coeffs, times)

        y0 = self.initial(self.X.evaluate(times[0])) # [batch_size, hidden_dim]
        z = torchcde.cdeint(X=self.X,
                            func=self.vector_field,
                            z0=y0,
                            t=times,
                            method='euler',
                            ) # [batch_size, time_steps, hidden_dim]

        # Reconstruction
        predicted_batch = self.decoder(z) # [batch_size, time_steps, output_dim]

        # Classification
        lengths = mask.sum(dim=1).long()
        last_idx = lengths - 1
        z_final = z[torch.arange(z.size(0)), last_idx, :]
        logits = self.classifier(z_final)

        return predicted_batch, logits