"""
Table of Contents
- get_gradient_norm: This measures the overall strength of gradients across the model.
- get_update_norm: Tracks how much the model parameters change per update.
- get_
"""

def get_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.detach().data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5  # sqrt(sum of squares)
        

def get_update_norm(model, prev_params):
    total_norm = 0.0
    for param, prev_param in zip(model.parameters(), prev_params):
        if param.grad is not None:
            update_norm = (param.detach().data - prev_param).norm(2)  # L2 norm
            total_norm += update_norm.item() ** 2
    return total_norm ** 0.5

