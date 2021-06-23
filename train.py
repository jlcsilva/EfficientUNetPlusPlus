import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from metrics import dice_loss
from eval import eval_net

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CoronaryArterySegmentationDataset, RetinaSegmentationDataset
from torch.utils.data import DataLoader
from kornia.losses import focal_loss

import segmentation_models_pytorch.segmentation_models_pytorch as smp

def train_net(net,
              device,
              training_set,
              validation_set,
              dir_checkpoint,
              epochs=150,
              batch_size=2,
              lr=0.001,
              save_cp=True,
              img_scale=1,
              n_classes=3,
              n_channels=3, 
              augmentation_ratio = 0):

    train = training_set 
    val = validation_set
    n_train = len(train)
    n_val = len(val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Sets the effective batch size according to the batch size and the data augmentation ratio
    batch_size = (1 + augmentation_ratio)*batch_size

    # Prepares the summary file
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Augmentation ratio: {augmentation_ratio}
    ''')

    # Choose the optimizer and scheduler 
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, epochs//3, gamma=0.1, verbose=True)

    # Train loop
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                # DataLoaders return lists of tensors. TODO: Concatenate the lists inside the DataLoaders
                imgs = torch.cat(imgs, dim = 0)
                true_masks = torch.cat(true_masks, dim = 0)

                assert imgs.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)                         
                
                # Compute loss
                loss = focal_loss(masks_pred, true_masks.squeeze(1), alpha=0.25, gamma = 2, reduction='mean').unsqueeze(0)
                loss += dice_loss(masks_pred, true_masks.squeeze(1), True, k = 0.75)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0]//(1 + augmentation_ratio))
                global_step += 1
                if global_step % (n_train // (batch_size / (1 + augmentation_ratio))) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        
                        try:
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        except:
                            pass
                    
                    epoch_score = eval_net(net, train_loader, device)
                    val_score = eval_net(net, val_loader, device)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if n_classes > 1:
                        logging.info('Validation loss: {}'.format(val_score))
                        writer.add_scalar('Generalized dice loss/train', epoch_score, global_step)
                        writer.add_scalar('Generalized dice loss/test', val_score, global_step)
                    else:
                        logging.info('Validation loss: {}'.format(val_score))
                        writer.add_scalar('Dice loss/train', epoch_score, global_step)
                        writer.add_scalar('Dice loss/test', val_score, global_step)       
        scheduler.step()         
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='EfficientUNet++ train script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, help='Specifies the dataset to be used', dest='dataset', required=True)
    parser.add_argument('-ti', '--training-images-dir', type=str, default=None, help='Training images directory', dest='train_img_dir')
    parser.add_argument('-tm', '--training-masks-dir', type=str, default=None, help='Training masks directory', dest='train_mask_dir')
    parser.add_argument('-vi', '--validation-images-dir', type=str, default=None, help='Validation images directory', dest='val_img_dir')
    parser.add_argument('-vm', '--validation-masks-dir', type=str, default=None, help='Validation masks directory', dest='val_mask_dir')
    parser.add_argument('-enc', '--encoder', metavar='ENC', type=str, default='timm-efficientnet-b0', help='Encoder to be used', dest='encoder')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=150, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', type=str, default=False, help='Load model from a .pth file', dest='load')
    parser.add_argument('-s', '--scale', metavar='S', type=float, default=1, help='Downscaling factor of the images', dest='scale')
    parser.add_argument('-a', '--augmentation-ratio', metavar='AR', type=int, default=0, help='Number of augmentation to be generated for each image in the dataset', dest='augmentation_ratio')
    parser.add_argument('-c', '--dir_checkpoint', type=str, default='checkpoints/', help='Directory to save the checkpoints', dest='dir_checkpoint')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Number of classes
    if args.dataset == 'DRIVE':
        n_classes = 2
    elif args.dataset == 'Coronary':
        n_classes = 3

    # Instantiate EfficientUNet++ with the specified encoder
    net = smp.EfficientUnetPlusPlus(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=n_classes)

    # Freeze encoder weights
    net.encoder.eval()
    for m in net.encoder.modules():
        m.requires_grad_ = False

    # Distribute training over GPUs
    net = nn.DataParallel(net)

    # Load weights from file
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # Faster convolutions, but more memory usage
    #cudnn.benchmark = True

    # Instantiate datasets
    if args.dataset == 'DRIVE':
        training_set = RetinaSegmentationDataset(args.train_img_dir if args.train_img_dir is not None else 'DRIVE/training/images/', 
            args.train_mask_dir if args.train_mask_dir is not None else 'DRIVE/training/1st_manual/', args.scale, 
            augmentation_ratio = args.augmentation_ratio, crop_size=512)
        validation_set = RetinaSegmentationDataset(args.val_img_dir if args.val_img_dir is not None else 'DRIVE/validation/images/', 
            args.val_mask_dir if args.val_mask_dir is not None else 'DRIVE/validation/1st_manual/', args.scale)
        dataset_class = RetinaSegmentationDataset
    elif args.dataset == 'Coronary':
        training_set = CoronaryArterySegmentationDataset(args.train_img_dir if args.train_img_dir is not None else 'Coronary/train/imgs/', 
            args.train_mask_dir if args.train_mask_dir is not None else 'Coronary/train/masks/', args.scale, 
            augmentation_ratio = args.augmentation_ratio, crop_size=512)
        validation_set = CoronaryArterySegmentationDataset(args.val_img_dir if args.val_img_dir is not None else 'Coronary/val/imgs/', 
            args.val_mask_dir if args.val_mask_dir is not None else 'Coronary/val/masks/', args.scale, mask_suffix='a')
        dataset_class = RetinaSegmentationDataset
    else:
        print("Invalid dataset")
        exit()

    try:
        train_net(net=net, 
                  device=device,
                  training_set=training_set,
                  validation_set=validation_set,
                  dir_checkpoint=args.dir_checkpoint,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  img_scale=args.scale,
                  n_classes=n_classes,
                  n_channels=3,
                  augmentation_ratio = args.augmentation_ratio)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
