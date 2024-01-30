import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FixMatch(nn.Module):
    def __init__(self, args, temperature, threshold):
        super().__init__()
        self.device = args.device
        self.mu = args.mu
        self.T = temperature
        self.threshold = threshold

    def forward(self, inputs_uw, inputs_us, model):

        inputs_uw, inputs_us = inputs_uw.cuda(), inputs_us.cuda()
        outputs_u = model(inputs_uw)[0]
        targets_u = torch.softmax(outputs_u, dim=1)

        max_p, p_hat = torch.max(targets_u, dim=1)

        mask = max_p.ge(self.threshold).float()
        outputs = model(inputs_us)[0]

        ssl_loss = (F.cross_entropy(outputs, p_hat, reduction='none') * mask).mean()
        return ssl_loss


class ADSH(nn.Module):
    def __init__(self, args, temperature, threshold):
        super().__init__()
        self.mu = args.mu
        self.T = temperature
        self.threshold = threshold

    def forward(self, inputs_uw, inputs_us, model, score):
        inputs_uw, inputs_us = inputs_uw.cuda(), inputs_us.cuda()
        outputs_uw = model(inputs_uw)[0]
        probs = torch.softmax(outputs_uw, dim=1)

        rectify_prob = probs / torch.from_numpy(score).float().cuda()
        max_rp, rp_hat = torch.max(rectify_prob, dim=1)
        mask = max_rp.ge(1.0)

        outputs, _, strong_features = model(inputs_us, return_feature=True)
        strong_soft_labels = F.softmax(outputs)

        ssl_loss = (F.cross_entropy(outputs, rp_hat, reduction='none') * mask).mean()
        return ssl_loss, strong_features, strong_soft_labels
