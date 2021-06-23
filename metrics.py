import torch
import torch.nn.functional as F

"""
Implements metrics to be used as loss functions or performance evaluation criteria.
"""
def dice_loss(input: torch.FloatTensor, target: torch.LongTensor, use_weights: bool = False, k: int = 0, eps: float = 0.0001):
    """
    Returns the Generalized Dice Loss Coefficient of a batch associated to the input and target tensors. In case `use_weights` \
        is specified and is `True`, then the computation of the loss takes the class weights into account.

    Args:
        input (torch.FloatTensor): NCHW tensor containing the probabilities predicted for each class.
        target (torch.LongTensor): NCHW one-hot encoded tensor, containing the ground truth segmentation mask. 
        use_weights (bool): specifies whether to use class weights in the computation or not.
        k (int): weight for pGD function. Default is 0 for ordinary dice loss.
    """
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # Multiple class case
    n_classes = input.size()[1]
    if n_classes != 1:
        # Convert target to one hot encoding
        target = F.one_hot(target, n_classes).squeeze()
        if target.ndim == 3:
            target = target.unsqueeze(0)
        target = torch.transpose(torch.transpose(target, 2, 3), 1, 2).type(torch.FloatTensor).cuda().contiguous()
        input = torch.softmax(input, dim=1)
    else:
        input = torch.sigmoid(input)   

    class_weights = None
    for i, c in enumerate(zip(input, target)):
        if use_weights:
            class_weights = torch.pow(torch.sum(c[1], (1,2)) + eps, -2)
        s = s + __dice_loss(c[0], c[1], class_weights, k=k)

    return s / (i + 1)

def __dice_loss(input: torch.FloatTensor, target: torch.LongTensor, weights: torch.FloatTensor = None, k: int = 0, eps: float = 0.0001):
    """
    Returns the Generalized Dice Loss Coefficient associated to the input and target tensors, as well as to the input weights,\
    in case they are specified.

    Args:
        input (torch.FloatTensor): CHW tensor containing the classes predicted for each pixel.
        target (torch.LongTensor): CHW one-hot encoded tensor, containing the ground truth segmentation mask. 
        weights (torch.FloatTensor): 2D tensor of size C, containing the weight of each class, if specified.
        k (int): weight for pGD function. Default is 0 for ordinary dice loss.
    """  
    n_classes = input.size()[0]

    if weights is not None:
        for c in range(n_classes):
            intersection = (input[c] * target[c] * weights[c]).sum()
            union = (weights[c] * (input[c] + target[c])).sum() + eps
    else:
        intersection = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps    

    gd = (2 * intersection.float() + eps) / union.float()
    return 1 - (gd / (1 + k*(1-gd)))