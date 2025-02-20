
import torch
from torch.utils.data import random_split, DataLoader
import os 


root_data = os.path.dirname(os.path.abspath(__file__))
__accepted_dataset__ = [
   'resnet18_layer13_imagenet1ktrain_goldfishonly',
]


num_workers = 4
train_val_ratio = 0.85 # 1 = all training, 0 = all validation


def get_DataSet(dataset_name):
    """
    Args:
        train (str): Dataset name for training model.
        val (str): Dataset name for validating model.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in __accepted_dataset__:
        raise ValueError('Dataset not supported')

    # Load dataset
    dataset_path = os.path.join(root_data, f"{dataset_name}.pt")
    dataset = torch.load(dataset_path)

    # Partition into training & validation
    train_size = int(train_val_ratio * len(dataset))  
    val_size = len(dataset) - train_size 
    dataset_train, dataset_val = random_split(dataset, [train_size, val_size])

    return dataset_train, dataset_val


def get_DataLoader(batch_size: int, 
                   dataset_name: str):
    """
    Args:
        batch_size (int)
        dataset_name (str)
    """
    train_dataset, val_dataset = get_DataSet(dataset_name)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader

