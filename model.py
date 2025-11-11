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
    def __init__(self, input_shape=(1, 42, 42), latent_channel=128, token_count=2, embed_dim=128): # Actual token count is token_count * token_count
        super().__init__()
        self.features = Featuriser(input_channel=input_shape[0], output_channel=latent_channel)
        self.token_count = token_count
        self.cnn_out_shape = self._get_conv_output_shape(input_shape)
        self.proj = nn.Sequential(
            nn.Linear(latent_channel * (self.cnn_out_shape[-1] // token_count) * (self.cnn_out_shape[-2] // token_count), embed_dim),
            nn.LayerNorm(embed_dim), 
            nn.ReLU()
        )
        self.positional_embedding = RotaryPositionalEmbedding(embed_dim, max_seq_len=token_count*token_count)
        self.attn = SelfAttention(embed_dim=embed_dim, num_heads=4, ff_dim=embed_dim*4, dropout=0.1)
        self.classifier = Classifier(input_features=embed_dim, num_classes=num_classes)
        

    def forward(self, x):
        # x: [B, 1, 42, 42]
        x = self.features(x)
        # x: [B, latent_channel, H', W']

        x = self.split_latent(x, n = self.token_count)
        # x: [B, n*n, latent_channel, H'//n, W'//n]

        x = x.view(x.shape[0], x.shape[1], -1)
        # x: [B, n*n, latent_channel * H'//n * W'//n]
        
        tokens = self.proj(x)
        # x: [B, n*n, embed_dim]

        tokens = self.positional_embedding(tokens)
        # x: [B, n*n, embed_dim]

        x = self.attn(tokens)
        # x: [B, n*n, embed_dim]

        x_pooled = x.mean(dim=1)
        # x: [B, embed_dim]

        x = self.classifier(x_pooled)
        return x
    
    def split_latent(self, x, n):
        B, C, H, W = x.shape
        assert H % n == 0 and W % n == 0, "H and W must be divisible by n"

        # Step 1: reshape to split height and width into n parts
        x = x.view(B, C, n, H // n, n, W // n)  # (B, latent_channel, n, H//n, n, W//n)
        
        # Step 2: rearrange dimensions to bring patches together
        x = x.permute(0, 2, 4, 1, 3, 5)         # (B, n, n, latent_channel, H//n, W//n)
        
        # Step 3: flatten n×n patches into one dimension
        x = x.reshape(B, n * n, C, H // n, W // n)  # (B, n*n, latent_channel, H//n, W//n)
        
        return x
    
    def _get_conv_output_shape(self, input_shape):
        """Pass a dummy input to conv layers to compute flatten size"""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.features(x)
            return x.shape
    
class Classifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()

        # Create a rotation matrix.
        self.rotation_matrix = torch.zeros(d_model, d_model, device=torch.device("cuda"))
        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = torch.cos(i * j * 0.01)

        # Create a positional embedding matrix.
        self.positional_embedding = torch.zeros(max_seq_len, d_model, device=torch.device("cuda"))
        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = torch.cos(i * j * 0.01)

    def forward(self, x):
        """
        Args:
            x: A tensor of shape (batch_size, seq_len, d_model).

        Returns:
            A tensor of shape (batch_size, seq_len, d_model).
        """

        # Add the positional embedding to the input tensor.
        x += self.positional_embedding

        # Apply the rotation matrix to the input tensor.
        x = torch.matmul(x, self.rotation_matrix)

        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim or embed_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim or embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.attn_norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        return x
    
class Featuriser(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(input_channel, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout(0.05),

            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            ConvBlock(64, output_channel),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        x = self.features(x)
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