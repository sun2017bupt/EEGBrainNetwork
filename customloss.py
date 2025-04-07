import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        n = log_probs.size(0)
        target_index = target.long()
        # print(f"log probs:{log_probs.shape} input shape:{input.shape}")
        # if len(input.shape) == 1:
        #     smooth_loss = -((1 - self.smoothing) * log_probs[target_index].mean() 
        #                 + self.smoothing /2 * log_probs.mean())
        # else:
        smooth_loss = -((1 - self.smoothing) * log_probs[torch.arange(n),target_index].sum(-1).mean()
                        + self.smoothing / n * log_probs.sum(-1).mean())
        return smooth_loss