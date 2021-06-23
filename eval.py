import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import dice_loss

def eval_net(net, loader, device, n_classes=3):
    net.eval()
    mask_type = torch.float32 if n_classes == 1 else torch.long
    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'][0], batch['mask'][0]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if n_classes > 1:
                true_masks = true_masks.squeeze(1)
                tot += dice_loss(mask_pred, true_masks, use_weights=True).item()
            else:
                tot += dice_loss(mask_pred, true_masks, use_weights=False).item()
            pbar.update()

    net.train()
    return tot / n_val