import numpy as np

from colour_transfer_MKL.color_transfer_MKL import color_transfer_MKL
from WB_sRGB.WB_sRGB_Python.classes.WBsRGB import WBsRGB

def image_asserts(image: np.array):
    assert image.dtype == np.uint8, "Image must be 8bit"
    assert len(image.shape) == 3, "Input must be a BGR color image"
    assert image.shape[2] == 3, "Image have 3 channels"

############## AUTO WB PARAMS ################
upgraded_model = 0
# use gamut_mapping = 1 for scaling, 2 for clipping (our paper's results
# reported using clipping). If the image is over-saturated, scaling is
# recommended.
gamut_mapping = 2

# processing
# create an instance of the WB model
wbModel = WBsRGB(gamut_mapping=gamut_mapping, upgraded=upgraded_model)


def wb_autocorrect(image: np.ndarray):
    """Apply sRGB auto white balance
    
    Args:
        image: np.ndarray -- **BGR** input image

    Returns:
        wb_image: np.ndarray -- *BGR* auto-WB image
    """
    image_asserts(image)

    autowb = wbModel.correctImage(image)

    return np.uint8(np.clip(autowb * 255., 0, 255))


def wb_from_ref(source: np.ndarray, target: np.ndarray):
    """Perform color transfer from one sRGB image to another
    
    Args:
        source: np.ndarray -- Image to be recolored
        target: np.ndarray -- Image with target color scheme

    Returns:
        trans: np.array -- Source image with changed colors
    """
    image_asserts(source)
    image_asserts(target)

    source = source / 255.
    target = target / 255.

    trans = color_transfer_MKL(source, target)

    return np.uint8(np.clip(trans * 255., 0, 255))
