import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f'enc_{i+1:d}').parameters():
                param.requires_grad = False


    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i+1:d}')
            results.append(func(results[-1]))
        return results[1:]



class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.VGG = VGG16()

    def calculate_loss(self, generated_images, target_images, device):
        self.VGG = self.VGG.to(device)

        generated_features = self.VGG(generated_images)
        target_features = self.VGG(target_images)

        perceptual_loss = 0
        perceptual_loss += torch.mean((target_features[0] - generated_features[0]) ** 2)
        perceptual_loss += torch.mean((target_features[1] - generated_features[1]) ** 2)
        perceptual_loss += torch.mean((target_features[2] - generated_features[2]) ** 2)
        perceptual_loss /= 3
        return perceptual_loss
    

class SkeletLoss(nn.Module):
    def __init__(self):
        super(SkeletLoss, self).__init__()
        self.transforms_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.perceptual_loss = PerceptualLoss()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = 1.
        self.perceptual_weight = 0.01

    def forward(self, preds, targets):
        
        mse = self.mse_loss(preds, targets) * self.mse_weight
        
        normalize_preds = (preds / 2 + 0.5).clamp(0, 1)
        normalize_targets = (targets / 2 + 0.5).clamp(0, 1)
        normalize_preds = self.transforms_norm(normalize_preds.repeat(1, 3, 1, 1))
        normalize_targets = self.transforms_norm(normalize_targets.repeat(1, 3, 1, 1))
        percep = self.perceptual_loss.calculate_loss(normalize_preds, normalize_targets, preds.device) * self.perceptual_weight

        return mse + percep
    
