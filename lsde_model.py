import os
import sys
import torch
import torch.nn as nn
import torchcde
import torchsde
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utilities import LipSwish

class LSDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_classes, controlled_path=True):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"

        self.controlled_path = controlled_path

        # f
        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_in = torch.nn.Linear(hidden_dim, hidden_dim)
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            LipSwish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

        # g
        self.noise_in = nn.Linear(1, hidden_dim)
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
        if self.controlled_path:
            Xt = self.X.evaluate(t)
            Xt = self.linear_X(Xt)
            z = self.emb(torch.cat([y, Xt], dim=-1))
        else:
            z = self.linear_in(y)

        z = self.f_net(z)
        return self.linear_out(z)

    def g(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1) # [batch, 1]
        tt = self.noise_in(t) # [batch, hidden_dim]
        return self.g_net(tt)

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