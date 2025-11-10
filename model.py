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

class Model(nn.Module):
    def _get_conv_output(self, shape):
        """Pass a dummy input to conv layers to compute flatten size"""
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.features(x)
            n_features = x.numel()  # total features for nn.Linear
        return n_features
    
    def __init__(self, input_shape=(1, 42, 42)):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout(0.05),

            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        # --- Automatically compute flattened feature size ---
        self._to_linear = self._get_conv_output(input_shape)
        
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),  # adapt based on your input size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
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