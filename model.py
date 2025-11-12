import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import split_into_char_images, is_char_correct_with_allowance
# Add your Neural Network here (if using NN)
char_classes = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "a": 10,
    "b": 11,
    "c": 12,
    "d": 13,
    "e": 14,
    "f": 15,
    "g": 16,
    "h": 17,
    "i": 18,
    "j": 19,
    "k": 20,
    "l": 21,
    "m": 22,
    "n": 23,
    "o": 24,
    "p": 25,
    "q": 26,
    "r": 27,
    "s": 28,
    "t": 29,
    "u": 30,
    "v": 31,
    "w": 32,
    "x": 33,
    "y": 34,
    "z": 35,
}
num_classes = len(char_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class SmallResNet(nn.Module):
    def __init__(self, num_classes=36, input_size=(32,32)):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.layer2 = ResidualBlock(32, 64, downsample=True)
        self.layer3 = ResidualBlock(64, 128, downsample=True)
        self.layer4 = ResidualBlock(128, 128, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class STN(nn.Module):
    def __init__(self, input_size=(42, 42)):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Compute flattened dimension dynamically
        self.fc_input_dim = self._get_flattened_size(input_size)

        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # Initialize to identity transform
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def _get_flattened_size(self, input_size):
        """Run a dummy forward pass to find flatten size"""
        x = torch.zeros(1, 1, *input_size)
        x = self.localization(x)
        return x.numel()

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

class Model(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.stn = STN()
        self.resnet = SmallResNet(num_classes)
    def forward(self, x):
        x = self.stn(x)
        x = self.resnet(x)
        return x

def infer(net, captcha):
    char_images = split_into_char_images(captcha)
    predictions = []
    net.to(device)
    with torch.no_grad():
        for ci in char_images:
            img_tensor = torch.tensor(ci, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Add batch and channel dimensions
            img_tensor = img_tensor.to(device)
            output = net.forward(img_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            item = predicted_class.item()
            keys = [k for k, v in char_classes.items() if v == item]
            predictions.append(keys[0] if keys else None)
    return predictions

def count_params(net, trainable=False):
    if trainable:
        count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    else:
        count = sum(p.numel() for p in net.parameters())
    return count

def check_correctness(predictions, ground_truth, with_allowance=False):
    if len(predictions) != len(ground_truth):
        return False
    for p, t in zip(predictions, ground_truth):
        if with_allowance:
            if not is_char_correct_with_allowance(p, t):
                return False
        else:
            if p != t:
                return False
    return True