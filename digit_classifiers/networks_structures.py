import torch


class LeNet5(torch.nn.Module):
    def __init__(self, num_classes: int = 20,
                 activation_idx: int = 0, final_out_idx: int = 0):
        """
        :param num_classes:         Number of classes of labels
        """
        super(LeNet5, self).__init__()
        self.activation_idx = activation_idx
        self.final_out_idx = final_out_idx

        # Input (28*28) -- Convolution --> C1 (6@28*28)
        self.c1_conv = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                                       padding=2)
        # C1 (6@28*28)  -- Pooling     --> S2 (6@14*14)
        self.s2_pool = torch.nn.MaxPool2d(kernel_size=2)
        # S2 (6@14*14)  -- Convolution --> C3 (16@10*10)
        self.c3_conv = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # C3 (16@10*10) -- Pooling     --> S4 (16@5*5)
        self.s4_pool = torch.nn.MaxPool2d(kernel_size=2)
        # S4 (16@5*5)   -- Convolution --> C5 (120@1*1)
        self.c5_conv = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # C5 (120)      -- FC          --> F6 (84)
        self.f6_fc = torch.nn.Linear(in_features=120, out_features=84)
        # F6 (84)       -- FC          --> Output (20)
        self.out_fc = torch.nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        if 0 == self.activation_idx:  # relu activation
            func = torch.nn.functional.relu
        elif 1 == self.activation_idx:  # HardTanh activation
            func = torch.tanh
        elif 2 == self.activation_idx:  # sigmoid activation
            func = torch.sigmoid
        elif 3 == self.activation_idx:  # leaky_relu activation
            func = torch.nn.functional.leaky_relu  # negative slope = 0.01
        else:  # 4 == self.activation_idx:  # elu activation
            func = torch.nn.functional.elu  # alpha=1.0

        x = func(self.c1_conv(x))
        x = self.s2_pool(x)
        x = func(self.c3_conv(x))
        # x = torch.nn.functional.relu(self.c3_conv(x))
        x = self.s4_pool(x)
        # x = torch.nn.functional.relu(self.c5_conv(x))
        x = func(self.c5_conv(x))
        x = torch.flatten(x, start_dim=1)  # flatten for FC layer
        x = func(self.f6_fc(x))
        # x = torch.nn.functional.relu(self.f6_fc(x))
        x = self.out_fc(x)

        if 0 == self.final_out_idx:  # no more functions
            pass
        else:  # 1 == self.final_out_idx:
            x = torch.nn.functional.softmax(x)

        return x
