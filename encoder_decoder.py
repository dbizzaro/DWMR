import torch
import torch.nn as nn


class BitEncoder(nn.Module):
    def __init__(
        self, 
        in_ch=1, 
        channels = (8, 16, 32, 32, 16), 
        out_grid_dim=11,  # better if divides img_grid_dim/4
        hidden_dim_fc=96, 
        n_bits=64, 
        temperature=1.0, 
        noise_magnitude=1.0, 
    ):
        super().__init__()
        self.temperature = temperature
        self.noise_magnitude = noise_magnitude
        self.n_bits = n_bits
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], 3, 1, 1, bias=False), nn.GroupNorm(channels[0], channels[0]), nn.ReLU(), nn.AvgPool2d(2,2),   # 88→44
            nn.Conv2d(channels[0], channels[1], 3, 1, 1, bias=False), nn.GroupNorm(channels[1], channels[1]), nn.ReLU(), nn.AvgPool2d(2,2),  # 44→22
            nn.Conv2d(channels[1], channels[2], 3, 1, 1, bias=False), nn.GroupNorm(channels[2], channels[2]), nn.ReLU(), nn.AvgPool2d(2,2),
            nn.Conv2d(channels[2], channels[3], 3, 1, 1, bias=False), nn.GroupNorm(channels[3], channels[3]), nn.ReLU(),
            nn.Conv2d(channels[3], channels[4], 3, 1, 1, bias=False), nn.GroupNorm(channels[4], channels[4]), nn.ReLU(),
            nn.AdaptiveAvgPool2d(out_grid_dim) # 11→out_grid_dim
        )
        self.fc = nn.Sequential(
            nn.Linear(channels[-1]*out_grid_dim**2, hidden_dim_fc), 
            nn.ReLU(), 
            nn.Linear(hidden_dim_fc, n_bits)
        )

    def forward(self, x, sample=False):
        feats = self.cnn(x)
        feats = feats.flatten(1)                          
        logits = self.fc(feats)
        if sample and self.noise_magnitude > 0.:  # use logistic noise for relaxed sampling
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            logistic = (u.log() - (1 - u).log())
            logistic = logistic * self.noise_magnitude
            y = torch.sigmoid((logistic + logits) / self.temperature)  # relaxed sample in [0,1]
        else:
            y = torch.sigmoid(logits/self.temperature)  # use probabilities directly
        y_hard = (y > 0.5).float()
        y_ste = y_hard.detach() + y - y.detach()  # straight-through estimator
        return logits, y, y_ste




class BitEncoderIceSlider(nn.Module):
    def __init__(
        self, 
        in_ch=1,
        channels= (32, 3), 
        out_grid_dim=8, 
        fc_head = False,
        n_bits=192, 
        temperature=1.0, 
        noise_magnitude=1.0, 
    ):
        super().__init__()
        self.temperature = temperature
        self.noise_magnitude = noise_magnitude
        self.n_bits = n_bits
        self.fc_head = fc_head
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], 4, 4, 0, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], 2, 2, 0, bias=True),  
        )
        self.fc = nn.Sequential(
            nn.Linear(channels[-1]*out_grid_dim**2, n_bits)
        )

    def forward(self, x, sample=False):
        feats = self.cnn(x)
        feats = feats.flatten(1)                          
        logits = self.fc(feats) if self.fc_head else feats
        if sample and self.noise_magnitude > 0.:  # use logistic noise for relaxed sampling
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            logistic = (u.log() - (1 - u).log())
            logistic = logistic * self.noise_magnitude
            y = torch.sigmoid((logistic + logits) / self.temperature)  # relaxed sample in [0,1]
        else:
            y = torch.sigmoid(logits/self.temperature)  # use probabilities directly
        y_hard = (y > 0.5).float()
        y_ste = y_hard.detach() + y - y.detach()  # straight-through estimator
        return logits, y, y_ste
    




class BitDecoder(nn.Module):
    def __init__(
        self,
        out_ch=1,
        channels=(8, 16, 32, 16, 16),
        in_grid_dim=11,     # must match encoder's out_grid_dim
        hidden_dim_fc=96,   # should match encoder.hidden_dim
        n_bits=64, 
    ):
        super().__init__()
        self.n_bits = n_bits
        self.channels = channels
        self.in_grid_dim = in_grid_dim
        self.hidden_dim_fc = hidden_dim_fc
        self.out_ch = out_ch
        self.fc = nn.Sequential(
            nn.Linear(n_bits, hidden_dim_fc),
            nn.ReLU(),
            nn.Linear(hidden_dim_fc, channels[-1] * in_grid_dim**2)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-2], 3, 1, 1, bias=False), nn.GroupNorm(channels[-2], channels[-2]), nn.ReLU(),
            nn.Conv2d(channels[-2], channels[-3], 3, 1, 1, bias=False), nn.GroupNorm(channels[-3], channels[-3]), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),           # 11 -> 22
            nn.Conv2d(channels[-3], channels[-4], 3, 1, 1, bias=False), nn.GroupNorm(channels[-4], channels[-4]), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),           # 22 -> 44
            nn.Conv2d(channels[-4], channels[-5], 3, 1, 1, bias=False), nn.GroupNorm(channels[-5], channels[-5]), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),           # 44 -> 88
            nn.Conv2d(channels[-5], out_ch, 3, 1, 1, bias=True)
        )

    def forward(self, z):
        h =  self.fc(z)
        h = h.view(h.shape[0], self.channels[-1], self.in_grid_dim, self.in_grid_dim)
        x_logits = self.dec(h)
        return x_logits
    


class BitDecoderIceslider(nn.Module):
    def __init__(
        self,
        out_ch=3,
        channels=(32, 3),
        in_grid_dim=8,     # must match encoder's out_grid_dim
        fc_head = True,
        n_bits=192, 
    ):
        super().__init__()
        self.n_bits = n_bits
        self.channels = channels
        self.in_grid_dim = in_grid_dim
        self.fc_head = fc_head
        self.out_ch = out_ch
        self.fc = nn.Sequential(
            nn.Linear(n_bits, channels[-1] * in_grid_dim**2)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(channels[-1], channels[-2], 2, 2, 0, bias=False), nn.BatchNorm2d(channels[-2]), nn.ReLU(),
            nn.ConvTranspose2d(channels[-2], out_ch, 4, 4, 0, bias=True)
        )

    def forward(self, z):
        z = z.view(z.shape[0], self.channels[-1], self.in_grid_dim, self.in_grid_dim)
        x_logits = self.dec(z)
        return x_logits
