from torch import nn

def create_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.GroupNorm(8, 32),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(32, 1, kernel_size=3, padding=1),
    )

    return model
