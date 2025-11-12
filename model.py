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
    def __init__(self, num_classes=36, input_size=(42,42)):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        
        # Compute the flattened size dynamically
        self._flattened_size = self._get_flattened_size(input_size)
        self.fc1 = nn.Linear(self._flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flattened_size(self, input_size):
        # Create a dummy tensor with batch size 1
        x = torch.zeros(1, 1, *input_size)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x.numel()  # Total number of elements

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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