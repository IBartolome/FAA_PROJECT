
from scipy import ndimage


def delete_background(img):

    tranf = ndimage.median_filter(img, size=10)

    tranf[(tranf > 0.6)] = 1
    tranf[(tranf <= 0.6)] = 0

    return tranf