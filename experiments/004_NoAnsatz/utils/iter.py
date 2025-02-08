from itertools import product

def generate_combinations(config):
    def flatten_dict(d, parent_key=''):
        """Recursively flatten dictionary, converting nested keys into dot notation."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v if isinstance(v, list) else [v]))
        return dict(items)

    flat_config = flatten_dict(config)
    keys, values = zip(*flat_config.items())
    combinations = [dict(zip(keys, vals)) for vals in product(*values)]
    
    # Convert flattened keys back into nested dictionary format
    def unflatten_dict(d):
        result = {}
        for k, v in d.items():
            keys = k.split('.')
            temp = result
            for key in keys[:-1]:
                temp = temp.setdefault(key, {})
            temp[keys[-1]] = v
        return result

    return [unflatten_dict(comb) for comb in combinations]