#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=1):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))      #先(1,0),(1,1)再(2,0),(2,1)...  前面为外循环，后面为内循环


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:        #ids是元组如(1000,0)
        im = resize_and_crop(Image.open(dir + str(id) + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale ,suffix):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask,suffix , scale)

    return zip(imgs_normalized, masks)

def get_imgs_and_masks_AE(ids, dir_img, dir_mask, scale ,suffix):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, suffix, scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    #imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask,suffix , scale)

    return zip(imgs_switched, masks)




def get_imgs_and_masks_AE_test(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '_predict_masks_epi.gif', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    #imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask,'_masks_epi.gif' , scale)

    return zip(imgs_switched, masks)

def get_val_imgs(ids, dir_img, dir_mask, scale ,suffix):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

   # masks = to_cropped_imgs(ids, dir_mask,suffix , scale)

    return imgs_normalized

def get_val_mask(ids, dir_img, dir_mask, scale ,suffix):
    """Return all the couples (img, mask)"""

#    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
 #   imgs_switched = map(hwc_to_chw, imgs)
 #   imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask,suffix , scale)

    return list(masks)
'''
def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_masks_epi.gif')
    return np.array(im), np.array(mask)
'''