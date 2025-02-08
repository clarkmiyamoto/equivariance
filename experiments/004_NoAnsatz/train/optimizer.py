import torch.optim as optim

def get_optim(name: str, model, lr, **options):
    name = name.lower()
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, **options)
    elif name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, **options)
    elif name == 'sgd':
        return optim.SGD(model.parameters(),lr=lr, **options)
    