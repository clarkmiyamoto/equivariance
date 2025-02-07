import torch

def sin_distortion(x_length: int,
                   y_length: int,
                   A_nm: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Sin distortion for creating deformation maps.

    Args:
    - x_length (int): Length of x-axis of image.
    - y_length (int): Length of y-axis of image.
    - A_nm (torch.Tensor): Square matrix of coefficients. Sets size of cut off.

    Returns:
    (torch.Tensor, torch.Tensor): Deformation maps for x and y coordinates.
    """
    if A_nm.shape[0] != A_nm.shape[1]:
        raise ValueError('A_nm must be square matrix.')

    A_nm = A_nm.float()

    # Create Coordinates
    x = torch.linspace(-1, 1, x_length, dtype=torch.float32)
    y = torch.linspace(-1, 1, y_length, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Create Diffeo
    x_pert = torch.linspace(0, 1, x_length, dtype=torch.float32)
    y_pert = torch.linspace(0, 1, y_length, dtype=torch.float32)

    n = torch.arange(1, A_nm.shape[0] + 1, dtype=torch.float32)
    x_basis = torch.sin(torch.pi * torch.outer(n, x_pert)).T
    y_basis = torch.sin(torch.pi * torch.outer(n, y_pert))

    perturbation = torch.matmul(x_basis, torch.matmul(A_nm, y_basis))

    x_map = X + perturbation
    y_map = Y + perturbation

    return x_map, y_map

def apply_transformation(image_tensor,
                         A_nm: torch.Tensor,
                         interpolation_type='bilinear'):
    """
    Wrapper of `sin_distortion`. Gets torch.tensor and returns the distorted
    torch.tensor according to A_nm.

    Args:
        image_tensor (torch.Tensor): Inputted image.
        A_nm (torch.Tensor): Characterizes diffeo according to `sin_distortion`.
        interpolation_type (str): Interpolation method ('bilinear' or 'nearest').

    Returns:
        image_tensor_deformed (torch.Tensor): Diffeo applied to `image_tensor`.
    """
    # Create deformation map
    x_length, y_length = image_tensor.shape[1:3]
    x_map, y_map  = sin_distortion(x_length, y_length, A_nm)

    return apply_flowgrid(image_tensor, x_map, y_map, interpolation_type=interpolation_type)


def apply_flowgrid(image_tensor, x_map, y_map, interpolation_type='bilinear'):
    # Stack and unsqueeze to form grid
    grid = torch.stack((y_map, x_map), dim=-1).unsqueeze(0).to(image_tensor.device)

    # Apply grid sample
    image_tensor_deformed = torch.nn.functional.grid_sample(image_tensor.unsqueeze(0),
                                                            grid,
                                                            mode=interpolation_type,
                                                            align_corners=True)

    return image_tensor_deformed.squeeze(0)