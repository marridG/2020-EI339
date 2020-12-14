import torch


class LeNet5(torch.nn.Module):
    def __init__(self, num_classes: int = 20):
        """
        :param num_classes:         Number of classes of labels
        """
        super(LeNet5, self).__init__()
        # Input (28*28) -- Convolution --> C1 (6@28*28)
        self.c1_conv = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                                       padding=2)
        # C1 (6@28*28)  -- Pooling     --> S2 (6@14*14)
        self.s2_pool = torch.nn.MaxPool2d(kernel_size=2)
        # S2 (6@14*14)  -- Convolution --> C3 (16@10*10)
        self.c3_conv = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # C3 (16@10*10) -- Pooling     --> S4 (16@5*5)
        self.s4_pool = torch.nn.MaxPool2d(kernel_size=2)
        # S4 (16@5*5)   -- FC          --> C5 (120)
        self.c5_fc = torch.nn.Linear(in_features=16 * 5 * 5, out_features=120)
        # C5 (120)      -- FC          --> F6 (84)
        self.f6_fc = torch.nn.Linear(in_features=120, out_features=84)
        # F6 (84)       -- FC          --> Output (20)
        self.out_fc = torch.nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.c1_conv(x))
        x = self.s2_pool(x)
        x = torch.nn.functional.relu(self.c3_conv(x))
        x = self.s4_pool(x)
        x = x.view(-1, 16 * 5 * 5)  # flatten for FC layer
        x = torch.nn.functional.relu(self.c5_fc(x))
        x = torch.nn.functional.relu(self.f6_fc(x))
        x = self.out_fc(x)

        return x
