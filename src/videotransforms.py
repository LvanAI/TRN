# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Implementation of transforms for grouped images"""
import math
import random
from collections import Sized

import numpy as np
from PIL import Image
from mindspore.dataset.transforms.validators import check_random_transform_ops
from mindspore.dataset.vision import Inter
from mindspore.dataset.vision.c_transforms import RandomHorizontalFlip


class GroupRandomHorizontalFlip:
    """Horizontal flip augmentation"""

    def __init__(self, prob=0.5):
        self.worker = RandomHorizontalFlip(prob)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupScale:
    """Rescales the grouped input PIL.Image to the given 'size'.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio.
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

        if not (isinstance(self.size, int) or (isinstance(self.size, Sized) and len(self.size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(self.size))

    def worker(self, img):
        """Resize the input PIL Image to the given size.
        Args:
            img (PIL Image): Image to be resized.

        Returns:
            PIL Image: Resized image.
        """
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)

            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), self.interpolation)

        return img.resize(self.size[::-1], self.interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]



class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):

        self.size =  size if not isinstance(size, int) else [size, size]

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        w, h = imgs[0].size
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        crop_imgs = [img.crop((i, j, i + tw, j + th)) for img in imgs]
        return crop_imgs


class RandomResizedCropVideo:
    def __init__(self, 
                size,
                p = 0.5,
                scale=(0.8, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),):

        self.size =  size if not isinstance(size, int) else [size, size]
        self.scale = scale
        self.ratio = ratio
        self.p = p
        self.interpolation = Inter.LINEAR

    def __call__(self, img_group):

        if random.random() < self.p:
            offset_h, offset_w, crop_h, crop_w = self.get_params(img_group, self.scale, self.ratio)
            crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
            ret_img_group = [img.resize((self.size[0], self.size[1]), self.interpolation) for img in crop_img_group]
                    
            return ret_img_group
        else:
            ret_img_group = [img.resize((self.size[0], self.size[1]), self.interpolation) for img in img_group]
                    
            return ret_img_group
    
    def get_params(self, imgs, scale, ratio):
        width, height = imgs[0].size
        area = height * width

        for _ in range(10):
            target_area = area * np.random.uniform(scale[0], scale[1])
            log_ratio = np.log(np.array(ratio))

            aspect_ratio = np.exp(
                np.random.uniform(log_ratio[0], log_ratio[1])
            )

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2

        return i, j, h, w


class VideoGaussianBlur:
    def __init__(self,p = 0.1, radius =2):
        super().__init__()
        
        self.p = p
        self.radius = 3
        

    def __call_(self, img_group):

        if random.random() < self.p:
            imgs = [img.gaussian_blur(self.radius) for img in img_group]
            return imgs
        else:
            return img_group



