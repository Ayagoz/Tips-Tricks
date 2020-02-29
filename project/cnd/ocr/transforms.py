import torch
import numpy as np


class ImageNormalization(object):
    def __call__(self, image):
        return image / 255.


class ToTensor(object):
    def __call__(self, image):
        image = image.astype(np.float32)
        return torch.from_numpy(image)

def get_transforms(image_size):
    transform = # USE COMPOSE TO APPLY ALL YOUR TRANSFORMS
    return transform
