# coding: utf-8 -*-
# (Commentaires de licence et d'auteur - laissés inchangés)

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

# from .resblock import ResBlock  # Si vous utilisez des ResBlocks, décommentez cette ligne

# from .attention_mechanisms import AttentionGateModule  # Décommentez si vous implémentez l'attention
# from .attention_mechanisms import ContextAttentionBlock


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 dropout=None, batch_norm=False, padding=1,
                 kernel_size=3, stride=1, dilation=1):
        """
        (convolution => [BN] => ReLU) * 2 + DropOut [optional]

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param mid_channels: number of middle channels, if None, mid-channels = out_channels, default is None
        :type mid_channels: int
        :param dropout: value of the DropOut probability, if None, Drop-Out is not applied, default is None
        :type dropout: float
        :param batch_norm: apply BatchNorm2d after each convolution layer, default is False
        :type batch_norm: bool
        :param padding:  padding value, default = 1
        :type padding: int
        :param kernel_size: size of the convolution kernel, default = 3
        :type kernel_size: int
        :param stride:  stride value, default = 1
        :type stride: int
        :param dilation: dilation value, default = 1
        :type dilation: int
        """
        super(DoubleConv, self).__init__()  # Correction:  Appeler super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Définition séquentielle des couches du bloc DoubleConv
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True),  # Activation ReLU après la première convolution

            nn.BatchNorm2d(mid_channels) if batch_norm else nn.Identity(),  # BatchNorm optionnelle
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True),  # Activation ReLU après la deuxième convolution

            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(), # BatchNorm optionnelle

            nn.Dropout(p=dropout) if dropout else nn.Identity()  # Dropout optionnel
        )

    def forward(self, x):
        """
        Passe avant à travers le bloc DoubleConv.

        :param x: Entrée du bloc (tenseur)
        :return: Sortie du bloc (tenseur)
        """
        return self.double_conv(x)



class Down2C(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=False, batch_norm=False):
        """
        Downscaling with maxpool then double conv block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param mid_channels: number of middle channels, if None, mid-channels = out_channels, default is None
        :type mid_channels: int
        :param dropout: value of the DropOut probability, if None, Drop-Out is not applied, default is None
        :type dropout: float
        :param batch_norm: apply BatchNorm2d after each convolution layer, default is False
        :type batch_norm: bool
        """
        super(Down2C, self).__init__()  # Correction:  Appeler super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Max pooling 2x2 pour réduire la résolution spatiale
            DoubleConv(in_channels, out_channels, mid_channels,
                       dropout=dropout, batch_norm=batch_norm)  # Bloc DoubleConv
        )

    def forward(self, x):
        """
        Passe avant à travers le bloc Down2C.
        :param x: Entrée du bloc
        :return: Sortie du bloc
        """
        return self.maxpool_conv(x)



class Up2C(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Upscaling (soit par upsampling bilinéaire, soit par convolution transposée)
        suivi d'un bloc DoubleConv.

        :param in_channels: Nombre de canaux d'entrée (doit être la somme des canaux des deux entrées)
        :param out_channels: Nombre de canaux de sortie
        :param bilinear: Si True, utilise l'upsampling bilinéaire.  Sinon, utilise ConvTranspose2d.
        """
        super(Up2C, self).__init__()  # Correction: Appeler super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # Upsampling bilinéaire.  Redimensionne l'image, puis applique une convolution pour ajuster les canaux.
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)  # in_channels // 2 pour mid_channels
        else:
            # Convolution transposée pour l'upsampling.  Augmente la résolution et réduit les canaux.
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        """
        Passe avant pour le bloc Up2C.

        :param x1: Entrée de la partie montante (upsampling)
        :param x2: Entrée de la partie descendante (skip connection)
        :return: Sortie du bloc
        """
        x1 = self.up(x1)  # Upsampling de x1

        # Gestion des différences de taille entre x1 (upsamplé) et x2 (skip connection)
        # Calcul des différences de taille en hauteur et largeur
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Padding de x1 pour qu'il ait la même taille que x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concaténation de x1 (upsamplé et paddé) et x2 (skip connection)
        x = torch.cat([x2, x1], dim=1)  # Concaténation selon la dimension des canaux (dim=1)
        return self.conv(x)  # Application du bloc DoubleConv



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Couche de convolution finale pour produire la sortie.

        :param in_channels: Nombre de canaux d'entrée
        :param out_channels: Nombre de canaux de sortie (nombre de classes)
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Convolution 1x1

    def forward(self, x):
        """
        Passe avant pour la couche de sortie.
        :param x: Entrée
        :return: Sortie (prédiction)
        """
        # return self.conv(x)
        return F.sigmoid(self.conv(x))  #removed sigmoid

class UNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, n_classes, dropout=None, batch_norm=False, bilinear=True, aspp = False, spatial_attention_3d = False):
        """
        Architecture U-Net complète.
            
        :param input_channels:  number of input channels of the image (ex: 1 for grey images, 3 for RGB images)
        :type input_channels: int
        :param hidden_channels: list of the number of channels of each hidden layer
        :type hidden_channels: list of int
        :param n_classes: number of output channels, equivalent to the number of classes, default 1
        :type n_classes: int
        :param dropout:  value of the DropOut probability, if None, Drop-Out is not applied, default is None
        :type dropout: float
        :param batch_norm: apply BatchNorm2d after each convolution layer, default is False
        :type batch_norm: bool
        :param bilinear: it True ,default is True
        :type bilinear: bool
        
        """
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Couche d'entrée
        self.inc = DoubleConv(input_channels, hidden_channels[0], dropout=dropout, batch_norm=batch_norm)

        # Partie descendante (Encoder)
        self.downs = nn.ModuleList()
        for i in range(len(hidden_channels) - 1):
            self.downs.append(
                Down2C(hidden_channels[i], hidden_channels[i+1], dropout=dropout, batch_norm=batch_norm)
            )

        # Partie montante (Decoder) + Skip Connections
        self.ups = nn.ModuleList()
        factor = 2 if bilinear else 1

        for i in range(len(hidden_channels) - 1, 0, -1):
            self.ups.append(
                Up2C(hidden_channels[i] , hidden_channels[i-1] // factor, bilinear)
            )
        # Couche de sortie
        self.outc = OutConv(hidden_channels[0], n_classes)
        self.sigmoid = nn.Sigmoid() # Added sigmoid

    def forward(self, x):
        """
        Passe avant complète du réseau U-Net.

        :param x: Image d'entrée
        :return: Masque de segmentation prédit
        """
        x1 = self.inc(x)  # Première couche de convolution
        x_downs = [x1]
        for down in self.downs:
            x_downs.append(down(x_downs[-1]))  # Partie descendante (encoder)

        x = x_downs[-1]

        for i, up in enumerate(self.ups):
            x = up(x, x_downs[-(i + 2)])  # Partie montante (decoder) avec skip connections

        logits = self.outc(x)  # Couche de sortie
        return self.sigmoid(logits)