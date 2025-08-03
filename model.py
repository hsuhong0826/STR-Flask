# model.py

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from pathlib import Path
import tempfile

# ======================== 模型架構 ========================

class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class FirstGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstGenerator, self).__init__()
        self.down1 = self.conv_block(in_channels, 64, True)
        self.down2 = self.conv_block(64, 128, True)
        self.down3 = self.conv_block(128, 256, True)
        self.down4 = self.conv_block(256, 512, True)
        self.down5 = self.conv_block(512, 512, True)
        self.down6 = self.conv_block(512, 512, True)
        self.dropout = nn.Dropout(0.5)
        self.up1 = self.deconv_block(512, 512, True)
        self.up2 = self.deconv_block(1024, 512, True)
        self.up3 = self.deconv_block(1024, 256, True)
        self.up4 = self.deconv_block(512, 128, True)
        self.up5 = self.deconv_block(256, 64, True)
        self.up6 = self.deconv_block(128, 64, True)
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, res_block=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if res_block:
            layers.append(ResNetBlock(out_channels))
        return nn.Sequential(*layers)

    def deconv_block(self, in_channels, out_channels, res_block=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if res_block:
            layers.append(ResNetBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.dropout(self.down4(d3))
        d5 = self.dropout(self.down5(d4))
        d6 = self.dropout(self.down6(d5))
        u1 = self.dropout(self.up1(d6))
        u2 = self.dropout(self.up2(torch.cat([u1, d5], dim=1)))
        u3 = self.dropout(self.up3(torch.cat([u2, d4], dim=1)))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        return self.final(u6)


class SecondGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SecondGenerator, self).__init__()
        self.down1 = self.conv_block(in_channels, 64, True)
        self.down2 = self.conv_block(64, 128, True)
        self.down3 = self.conv_block(128, 256, True)
        self.down4 = self.conv_block(256, 512, True)
        self.se_block1 = SEBlock(512)
        self.down5 = self.conv_block(512, 512, True)
        self.down6 = self.conv_block(512, 512, True)
        self.dropout = nn.Dropout(0.5)
        self.up1 = self.deconv_block(512, 512, True)
        self.up2 = self.deconv_block(1024, 512, True)
        self.se_block2 = SEBlock(512)
        self.up3 = self.deconv_block(1024, 256, True)
        self.up4 = self.deconv_block(512, 128, True)
        self.up5 = self.deconv_block(256, 64, True)
        self.up6 = self.deconv_block(128, 64, True)
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, res_block=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if res_block:
            layers.append(ResNetBlock(out_channels))
        return nn.Sequential(*layers)

    def deconv_block(self, in_channels, out_channels, res_block=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if res_block:
            layers.append(ResNetBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.dropout(self.down4(d3))
        d4 = self.se_block1(d4)
        d5 = self.dropout(self.down5(d4))
        d6 = self.dropout(self.down6(d5))
        u1 = self.dropout(self.up1(d6))
        u2 = self.dropout(self.up2(torch.cat([u1, d5], dim=1)))
        u2 = self.se_block2(u2)
        u3 = self.dropout(self.up3(torch.cat([u2, d4], dim=1)))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        return self.final(u6)


class DualGenerator(nn.Module):
    def __init__(self, g1, g2):
        super(DualGenerator, self).__init__()
        self.g1 = g1
        self.g2 = g2

    def forward(self, x):
        x = self.g1(x)
        x = self.g2(x)
        return x

# ======================== 雲端下載功能 ========================

def download_from_google_drive(url, save_path):
    """從 Google Drive 分享連結下載檔案"""
    import gdown
    gdown.download(url, save_path, quiet=False)

# ======================== 載入模型 ========================

def load_model(device='cpu'):
    # 設定模型檔案的暫存路徑
    tmp_dir = tempfile.gettempdir()
    g1_path = os.path.join(tmp_dir, "G1_epoch233_FID23.58.pth")
    g2_path = os.path.join(tmp_dir, "G2_epoch233_FID23.58.pth")

    # Google Drive 檔案下載網址（使用 gdown）
    g1_url = "https://drive.google.com/uc?id=18t0K8OFN5493FOUcsN76Chd9SabhF9W3"
    g2_url = "https://drive.google.com/uc?id=1Ggsv-ZQf2MVkiFKs1c6G1LASy33jL_zH"

    # 如果檔案不存在就下載
    if not os.path.exists(g1_path):
        download_from_google_drive(g1_url, g1_path)
    if not os.path.exists(g2_path):
        download_from_google_drive(g2_url, g2_path)

    # 載入模型
    g1 = FirstGenerator(1, 3)
    g2 = SecondGenerator(3, 3)
    g1.load_state_dict(torch.load(g1_path, map_location=device))
    g2.load_state_dict(torch.load(g2_path, map_location=device))
    g1.to(device).eval()
    g2.to(device).eval()
    return DualGenerator(g1, g2).to(device).eval()

# ======================== 推論函數 ========================

def run_inference(model, image, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 單通道
    ])
    image = image.convert('L')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)[0]
        output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)
        output_image = transforms.ToPILImage()(output_tensor.cpu())
    return output_image