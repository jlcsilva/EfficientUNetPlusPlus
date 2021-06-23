import argparse
import logging
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from utils.dataset import CoronaryArterySegmentationDataset, RetinaSegmentationDataset
import segmentation_models_pytorch.segmentation_models_pytorch as smp

from torch.backends import cudnn

def predict_img(net, dataset_class, full_img, device, scale_factor=1, n_classes=3):
    net.eval()

    img = torch.from_numpy(dataset_class.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if n_classes > 1:
            probs = torch.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        full_mask = tf(probs.cpu())   
    
    if n_classes > 1:
        return dataset_class.one_hot2mask(full_mask)
    else:
        return full_mask > 0.5


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, help='Specifies the dataset to be used', dest='dataset', required=True)
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of ouput images')
    parser.add_argument('--scale', '-s', type=float, help="Scale factor for the input images", default=1)
    parser.add_argument('-enc', '--encoder', metavar='ENC', type=str, default='timm-efficientnet-b0', help='Encoder to be used', dest='encoder')
    return parser.parse_args()


def get_output_filenames(args, dataset='DRIVE'):
    """
    Validates and/or computes the output path of the segmentation masks to be generated.
    """
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            if dataset == 'DRIVE':
                dest_f = f.replace("images", "mask")
                dest_f = dest_f.replace("_test.tif", ".png")
            elif dataset == 'Coronary':
                dest_f = f.replace("imgs", "pred")
            else:
                dest_f = f.replace(".", "mask.")
            dest_dir, _ = os.path.split(dest_f)
            # Create destination directory
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            out_files.append(dest_f)
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args, args.dataset)

    # Number of classes
    if args.dataset == 'DRIVE':
        n_classes = 2
        dataset_class = RetinaSegmentationDataset
    elif args.dataset == 'Coronary':
        n_classes = 3
        dataset_class = CoronaryArterySegmentationDataset
    else:
        print("Invalid dataset")
        exit()
    net = smp.EfficientUnetPlusPlus(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=n_classes)
    net = nn.DataParallel(net)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # faster convolutions, but more memory
    cudnn.benchmark = True
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    if device == 'cuda':
        score = torch.FloatTensor(1).cuda().zero_()
        weighted_score = torch.FloatTensor(1).cuda().zero_()
    else:
        score = torch.FloatTensor(1).zero_()
        weighted_score = torch.FloatTensor(1).zero_()

    
    for i, fn in enumerate(in_files):
        if args.dataset == 'DRIVE':
            orig_image = Image.open(fn)
            w, h = orig_image.size
            image = Image.new('RGB', (608, 608), (0, 0, 0))
            image.paste(orig_image, (0, 0))
        elif args.dataset == 'Coronary':
            image = Image.open(fn).convert(mode='RGB')
        mask = predict_img(net=net, dataset_class=dataset_class, full_img=image, scale_factor=args.scale, device=device)
        result = dataset_class.mask2image(mask)
        if args.dataset == 'DRIVE':
            result = result.crop((0, 0, w, h))
        result.save(out_files[i])
        logging.info("Mask saved to {}".format(out_files[i]))