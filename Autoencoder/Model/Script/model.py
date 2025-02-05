import torch.nn as nn

class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32,512),
            nn.ReLU(),
            nn.Linear(512, 64 * 9 * 9 * 9),
            nn.ReLU(),
            nn.Unflatten(1, (64, 9, 9, 9)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16), 
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1), 
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Conv3DAutoencoder()
model.eval()

