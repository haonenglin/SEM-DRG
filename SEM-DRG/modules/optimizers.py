import torch

def build_optimizer(args, model):
    visual = list(model.visual_extractor.parameters())
    visual_ids = {id(p) for p in visual}
    decoder = [p for p in model.parameters() if id(p) not in visual_ids and p.requires_grad]
    return torch.optim.AdamW([{'params': [p for p in visual if p.requires_grad], 'lr': 2e-05}, {'params': decoder, 'lr': 0.0002}], weight_decay=0.01, amsgrad=True)

def build_lr_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-07)
