import importlib


MODEL_REGISTRY = {
    'mlp_v2': 'model.mlp.v2',
}

def get_model(name: str):
    """
    Assume all architectures are defined with "Model" as the name

    ```
    Class Model(torch.nn):
        ...
    ```
    """
    name = name.lower()
    
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Model "{name}" not supported')

    module_path = MODEL_REGISTRY[name]
    class_name = 'Model' # Here's where to modify this assumption

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
