import torch.optim as optim

def get_scheduler(name: str, optimizer, **options):
    name = name.lower()

    if name == 'none':
        return None
    elif name == "steplr":
        return optim.lr_scheduler.StepLR(optimizer, **options)
    elif name == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **options)
    elif name == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **options)
    elif name == "onecyclelr":
        return optim.lr_scheduler.OneCycleLR(optimizer, **options)
        
    
        