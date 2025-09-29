import os
import sys
import torch
import torch.nn as nn
import torchcde
import torchsde
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utilities import LipSwish

class SDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_classes):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"

        # f
        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            LipSwish(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # g
        self.noise_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.g_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            LipSwish(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # forward
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.__dict__['X'] = torchcde.CubicSpline(self.coeffs, self.times)

    def f(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        yy = self.linear_in(torch.cat((t, y), dim=-1))
        return self.f_net(yy)

    def g(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        yy = self.noise_in(torch.cat((t, y), dim=-1))
        return self.g_net(yy)

    def forward(self, coeffs, times, mask):
        self.set_X(coeffs, times)

        y0 = self.initial(self.X.evaluate(times[0])) # [batch_size, hidden_dim]
        dt = 1 / len(times)
        z = torchsde.sdeint(sde=self, y0=y0, ts=times, dt=dt, method='euler')
        z = z.permute(1, 0, 2) # [batch_size, time_steps, hidden_dim]

        # Reconstruction
        predicted_batch = self.decoder(z) # [batch_size, time_steps, output_dim]

        # Classification
        lengths = mask.sum(dim=1).long()
        last_idx = lengths - 1
        z_final = z[torch.arange(z.size(0)), last_idx, :]
        logits = self.classifier(z_final)

        return predicted_batch, logits