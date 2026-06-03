import torch
import torch.nn as nn
import torch.nn.functional as F


def advLoss(source, target, device):

    sourceLabel = torch.ones(len(source))
    targetLabel = torch.zeros(len(target))
    Loss = nn.BCELoss()
    if device == 'cuda':
        Loss = Loss.cuda()
        sourceLabel, targetLabel = sourceLabel.cuda(), targetLabel.cuda()
    #print("sd={}\ntd={}".format(source, target))
    loss = Loss(source, sourceLabel) + Loss(target, targetLabel)
    return loss*0.5


def weightedAdvLoss(source, target, target_weight=None, device='cuda', eps=1e-6):
    source = source.view(-1)
    target = target.view(-1)
    if target_weight is None:
        return advLoss(source, target, device)

    sourceLabel = torch.ones_like(source)
    targetLabel = torch.zeros_like(target)
    target_weight = target_weight.detach().view(-1).to(target.device).clamp_min(0)
    if target_weight.numel() != target.numel():
        raise RuntimeError(
            "target_weight length {} does not match target length {}".format(
                target_weight.numel(), target.numel()
            )
        )

    source_loss = F.binary_cross_entropy(source, sourceLabel, reduction='mean')
    target_loss_each = F.binary_cross_entropy(target, targetLabel, reduction='none')
    target_loss = torch.sum(target_loss_each * target_weight) / (torch.sum(target_weight) + eps)
    return 0.5 * (source_loss + target_loss)


if __name__ == "__main__":
    pass
