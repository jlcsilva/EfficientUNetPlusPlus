from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
from torchvision import transforms

class TNetPolicy(object):
    """ 
        Applies the augmentation policy used in Jun et al's coronary artery segmentation T-Net
        https://arxiv.org/abs/1905.04197. As described by the authors, first they zoom-in or zoom-out at a 
        random ratio within +/- 20%. Then, the image is shifted, horizontally and vertically, at a random 
        ratio within +/- 20% of the image size (512 x 512). Then, the angiography is rotated between 
        +/- 30 degrees. The rotation angle is not larger because actual angiographies do not deviate much
        from this range. Finally, because the brightness of the angiography image can vary, the brightness is
        also changed within +/- 40% at random rates.
        Referred in the results as Aug1.
        Example:
        >>> policy = TNetPolicy()
        >>> transformed_img, transformed_mask = policy(image, mask)
    """
    def __init__(self, scale_ranges = [0.8, 1.2], img_size = [512, 512], translate = [0.2, 0.2], rotation = [-30, 30], brightness = 0.4):
        self.scale_ranges = scale_ranges 
        self.img_size = img_size
        self.translate = translate
        self.rotation = rotation
        self.brightness = brightness

    def __call__(self, image, mask = None):
            tf_mask = None
            tf_list = list() # List of transformation

            # Random zoom-in or zoom-out of -20% to 20%
            params = transforms.RandomAffine.get_params(degrees = [0, 0], translate = [0, 0], \
            scale_ranges = self.scale_ranges, img_size = self.img_size, shears = [0, 0])
            tf_image = transforms.functional.affine(image, params[0], params[1], params[2], params[3])
            if mask is not None:
                tf_mask = transforms.functional.affine(mask, params[0], params[1], params[2], params[3])

            # Random horizontal and vertical shift of -20% to 20%
            params = transforms.RandomAffine.get_params(degrees = [0, 0], translate = self.translate, \
            scale_ranges = [1, 1], img_size = self.img_size, shears = [0, 0])
            tf_image = transforms.functional.affine(tf_image, params[0], params[1], params[2], params[3])
            if mask is not None:
                tf_mask = transforms.functional.affine(tf_mask, params[0], params[1], params[2], params[3])

            # Random rotation of -30 to 30 degress
            angle = transforms.RandomRotation.get_params(self.rotation)
            tf_image = transforms.functional.rotate(tf_image, angle)
            if mask is not None:
                tf_mask = transforms.functional.rotate(tf_mask, angle)

            # Random brightness change of -40% to 40%
            tf = transforms.ColorJitter(brightness = self.brightness)
            tf_image = tf(tf_image)

            if mask is not None:
                return (tf_image, tf_mask)
            else:
                return tf_image
    
    def __repr__(self):
        return "TNet Coronary Artery Segmentation Augmentation Policy"

class RetinaPolicy(object):
    def __init__(self, scale_ranges = [1, 1.1], img_size = [512, 512], translate = [0.1, 0.1], rotation = [-20, 20], crop_dims = [480, 480], brightness = None):
        self.scale_ranges = scale_ranges 
        self.img_size = img_size
        self.translate = translate
        self.rotation = rotation
        self.brightness = brightness
        self.crop_dims = crop_dims

    def __call__(self, image, mask = None):
            tf_mask = None

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, self.crop_dims)
            tf_image = transforms.functional.crop(image, i, j, h, w)
            if mask is not None:
                tf_mask = transforms.functional.crop(mask, i, j, h, w) 

            # Random rotation of -20 to 20 degress
            angle = transforms.RandomRotation.get_params(self.rotation)
            tf_image = transforms.functional.rotate(tf_image, angle)
            if mask is not None:
                tf_mask = transforms.functional.rotate(tf_mask, angle)   

            # Random horizontal and vertical shift of -10% to 10%
            params = transforms.RandomAffine.get_params(degrees = [0, 0], translate = self.translate, \
            scale_ranges = [1, 1], img_size = self.img_size, shears = [0, 0])
            tf_image = transforms.functional.affine(tf_image, params[0], params[1], params[2], params[3])
            if mask is not None:
                tf_mask = transforms.functional.affine(tf_mask, params[0], params[1], params[2], params[3])

            # TODO: -10% to 10% may make more sense, due to the existance of images with black padding borders
            # Random zoom-in of 0% to 10%
            params = transforms.RandomAffine.get_params(degrees = [0, 0], translate = [0, 0], \
            scale_ranges = self.scale_ranges, img_size = self.img_size, shears = [0, 0])
            tf_image = transforms.functional.affine(tf_image, params[0], params[1], params[2], params[3])
            if mask is not None:
                tf_mask = transforms.functional.affine(tf_mask, params[0], params[1], params[2], params[3])

            # TODO: change brightness too
            # Random brightness change
            if self.brightness is not None:
                tf = transforms.ColorJitter(brightness = self.brightness)
                tf_image = tf(tf_image)

            if mask is not None:
                return (tf_image, tf_mask)
            else:
                return tf_image

    def __repr__(self):
        return "Retinal Vessel Segmentation Augmentation Policy"

class CoronaryPolicy(object):
    def __init__(self, scale_ranges = [1, 1.1], img_size = [512, 512], translate = [0.1, 0.1], rotation = [-20, 20], brightness = None):
        self.scale_ranges = scale_ranges 
        self.img_size = img_size
        self.translate = translate
        self.rotation = rotation
        self.brightness = brightness

    def __call__(self, image, mask = None):
            tf_mask = None
            # Random rotation of -20 to 20 degress
            angle = transforms.RandomRotation.get_params(self.rotation)
            tf_image = transforms.functional.rotate(image, angle)
            if mask is not None:
                tf_mask = transforms.functional.rotate(mask, angle)   

            # Random horizontal and vertical shift of -10% to 10%
            params = transforms.RandomAffine.get_params(degrees = [0, 0], translate = self.translate, \
            scale_ranges = [1, 1], img_size = self.img_size, shears = [0, 0])
            tf_image = transforms.functional.affine(tf_image, params[0], params[1], params[2], params[3])
            if mask is not None:
                tf_mask = transforms.functional.affine(tf_mask, params[0], params[1], params[2], params[3])

            # TODO: -10% to 10% may make more sense, due to the existance of images with black padding borders
            # Random zoom-in of 0% to 10%
            params = transforms.RandomAffine.get_params(degrees = [0, 0], translate = [0, 0], \
            scale_ranges = self.scale_ranges, img_size = self.img_size, shears = [0, 0])
            tf_image = transforms.functional.affine(tf_image, params[0], params[1], params[2], params[3])
            if mask is not None:
                tf_mask = transforms.functional.affine(tf_mask, params[0], params[1], params[2], params[3])

            # TODO: change brightness too
            # Random brightness change
            if self.brightness is not None:
                tf = transforms.ColorJitter(brightness = self.brightness)
                tf_image = tf(tf_image)

            if mask is not None:
                return (tf_image, tf_mask)
            else:
                return tf_image

    def __repr__(self):
        return "Coronary Artery Segmentation Augmentation Policy"