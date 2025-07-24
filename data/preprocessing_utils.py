import torch
import torch.nn.functional as F

def enlarge_object_mask(object_mask: torch.Tensor, kernel_size: int = 25):
    """
    Enlarge the object mask by a kernel size.
    """
    mask_for_dilation = object_mask.unsqueeze(0).unsqueeze(0).float()

    # A kxk kernel will expand the mask by (k-1)//2 pixel in each direction.
    padding = (kernel_size - 1) // 2

    # F.max_pool2d with a stride of 1 is equivalent to morphological dilation.
    dilated_mask_float = F.max_pool2d(mask_for_dilation,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding)

    # Convert the dilated mask back to a boolean tensor of shape (H, W)
    dilated_mask = dilated_mask_float.squeeze().to(torch.bool)

    return dilated_mask



