import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
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

class ImageSequenceClassifier(nn.Module):
    def __init__(self, input_shape=(1, 42, 42), latent_channel=128, embed_dim=128, num_heads=8, num_classes=num_classes):
        super().__init__()
        self.featuriser = Featuriser(input_channel=input_shape[0], output_channel=latent_channel) 
        self._to_linear = self._get_conv_output(input_shape)
        self.proj = nn.Sequential(
            nn.Linear(self._to_linear, embed_dim),
            nn.LayerNorm(embed_dim), 
            nn.ReLU()
        )
        self.attn = SelfAttention(dim=embed_dim, num_heads=num_heads)
        self.classifier = Classifier(input_features=embed_dim, num_classes=num_classes)


    def forward(self, x, mask=None):
        # x: [B, N, C, H, W], N is sequence length
        B, N, C, H, W = x.shape

        # x: [B * N, C, H, W]
        x = x.reshape(B*N, C, H, W)

        # x: [B * N, H * W * C_new], CNN feature extraction
        features = self.featuriser(x)  

        # x: [B, N, H * W * C_new], Recover sequence shape
        features = features.view(B, N, -1)

        # x: [B, N, E], Project to embedding dimension
        tokens = self.proj(features)

        # ---- Self-attention with mask ----
        # key_padding_mask: shape [B, N], True = PAD
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # invert: now True = pad
            key_padding_mask = key_padding_mask.to(x.device)

        attn_out = self.attn(tokens, key_padding_mask=key_padding_mask)

        # x: [B, N, Class], FC per token
        logits = self.classifier(attn_out) 
        
        # x: [B, N, Class]
        return logits
        
    def _get_conv_output(self, shape):
        """Pass a dummy input to conv layers to compute flatten size"""
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.featuriser(x)
            n_features = x.view(1, -1).size(1)
        return n_features

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

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim or dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim or dim*4, dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        attn_out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        x = self.attn_norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, input_channel=1):
        super(SpatialTransformer, self).__init__()

        # --- Localization network (small CNN) ---
        self.localization = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Compute the flattened size automatically using dummy input
        self.fc_input_dim = self._get_fc_input_dim(input_channel)
        
        # --- Regressor for 2x3 affine matrix ---
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the transformation as identity
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def _get_fc_input_dim(self, input_channel):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channel, 42, 42)  # use your input size
            out = self.localization(dummy)
            return out.view(1, -1).size(1)

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Apply the spatial transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

class Featuriser(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.stn = SpatialTransformer(input_channel=input_channel)  # ← Added STN
        
        self.features = nn.Sequential(
            ConvBlock(input_channel, 32),
            nn.MaxPool2d(2),
            nn.Dropout(0.05),

            ResidualConvBlock(32, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            ResidualConvBlock(64, output_channel),
            nn.Dropout(0.25),
        )

    def forward(self, x):
         # --- Apply STN first ---
        x = self.stn(x)
        
        x = self.features(x)
        # (B*N, C, H, W) -> (B*N, C*H*W)
        x = x.view(x.size(0), -1)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=8):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.gn1 = nn.GroupNorm(num_groups, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding)
        self.gn2 = nn.GroupNorm(num_groups, out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        return self.relu(out)

def infer(net, captcha):
    net.to(device)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()  # output shape: [1, H, W]
    ])
    char_imgs = [transform(ci) for ci in split_into_char_images(captcha)]  # list of [C,H,W]
    X = torch.stack(char_imgs).unsqueeze(0).to(device)  # shape [1, N, C, H, W]
    mask = torch.ones(1, len(char_imgs), dtype=torch.bool).to(device)
    logits = net(X, mask)  # [1, N, num_classes]
    preds = torch.argmax(logits, dim=-1).squeeze(0)
    predicted_chars = [k for i in preds for k, v in char_classes.items() if v == i]
    return predicted_chars

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