#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

def rgb2gray(img):
    """
    Conversion of a RGB image to a gray image by the Luminosity method
    Parameters
    ----------------
    img : The function receives a grayscale image of N * M dimensions
    Returns
    ----------------
    The function returns a grayscale image by the Luminosity method
    """

    img = img2uint8(img)
    R = 0.2125 * (np.double(img[:,:,0]))
    G = 0.7154 * (np.double(img[:,:,1]))
    B = 0.0721 * (np.double(img[:,:,2]))

    return np.uint8(R + G + B)

def img2uint8(img):
    '''
    Convert image to 8-bit format [0, 255].
    '''

    vmin = img.min()
    vmax = img.max()

    img = ((img - vmin) / (vmax - vmin)) * 255.0

    return np.uint8(img)

def plotresult(img_in, img_out, s1, s2):
    """
    Plot comparation images before and after Simplest Color Balance
    """

    fig, ((img_og, img_scb), (hist_og, hist_scb)) = plt.subplots(ncols=2, nrows=2, figsize=(12, 9))

    img_og.imshow(img2uint8(img_in), cmap='gray')
    img_og.set_title('Input original image')
    img_og.set_xticks([]), img_og.set_yticks([])
    hist_og.set_title('Input image histogram')
    plothist(hist(rgb2gray(img_in)), 'k', fig, hist_og)

    img_scb.imshow(img_out, cmap='gray')
    img_scb.set_title('Output Simplest Color Balance image \n $S_1$ = {} \n $S_2$ = {}'.format(s1, s2))
    img_scb.set_xticks([]), img_scb.set_yticks([])
    hist_scb.set_title('Output Simplest Color Balance image histogram')
    plothist(hist(rgb2gray(img_out)), 'k', fig, hist_scb)

def plothist(h, color, fig=None, ax=None):
    '''
    Show histogram of an image.

    Parameters
    ----------
    h : Array with a size (255, 1) that contains the intensity values of the
        pixels of the grayscale image.

    color : Color to display the histogram e.g. 'r'
    '''

    if fig == None:
        fig, ax = plt.subplots(1, 1)

    ax.stem(h, linefmt = '{}-'.format(color), markerfmt = 'none', basefmt = 'k-')
    ax.set_xlim(0, 255)
    ax.grid('on')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    cmap = plt.get_cmap('gray', 255)
    norm = mpl.colors.Normalize(vmin=0, vmax=255)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig.colorbar(sm, ax=ax, orientation='horizontal')

def hist(img):
    '''
    Get array of pixel intensity values in a grayscale image.

    img : Image in grayscale.
    '''

    if len(img.shape) > 2:
        raise ValueError('Image not in grayscale')

    h = np.zeros((256, 1))

    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            h[img[m, n]] += 1

    return h

def cumhist(img):
    '''
    Get cumulative histogram of grayscale image.
    '''

    h = hist(img)

    for i in range(1, len(h)):
        h[i] += h[i - 1]

    return h

def searchvmin(h, nx, ny, s1):
    '''
    Search min value with histogram.

    Parameters
    ----------
    h : Array with a size (255, 1) that contains the intensity values of the
        pixels of the grayscale image.
    nx : x-dimension of the image.
    ny : y-dimension of the image.
    s1 : Percentage of pixels saturated to the min value.

    Return
    ------
    vmin : New vmin.
    '''

    vmin = h.tolist().index(min(min(h)))
    n = nx * ny
    while h[vmin + 1] <= n * (s1 / 100):
        vmin += 1

    return int(vmin)

def searchvmax(h, nx, ny, s2):
    '''
    Search max value with histogram.

    Parameters
    ----------
    h : Array with a size (255, 1) that contains the intensity values of the
        pixels of the grayscale image.
    nx : x-dimension of the image.
    ny : y-dimension of the image.
    s2 : percentage of pixels saturated to the max value.

    Return
    ------
    vmax : New vmax.
    '''

    vmax = h.tolist().index(max(max(h))) - 1
    n = nx * ny
    while h[vmax - 1] > n * (1 - (s2 / 100)):
        vmax -= 1

    if vmax < h.tolist().index(max(max(h))) - 1:
        vmax += 1

    return int(vmax)

def saturate_rescale_pixels(img, vmin, vmax):
    '''
    The pixels are updated and the image is rescaled in [vmin vmax] by means of
    an affine transformation.
    '''

    np.putmask(img, img < vmin, vmin)
    np.putmask(img, img > vmax, vmax)

    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            img[m, n] = (img[m, n] - vmin) * (255 / (vmax - vmin))

    return img

def scb(img, s1, s2):
    '''
    Simplest Color Balance algoritm.

    img : RGB or Grayscale image.
    s1 : Percentage of pixels saturated to the min value.
    s2 : percentage of pixels saturated to the max value.

    Return
    ------
    out : Image with algorithm applied.
    '''

    if not (1 <= s1 <= 20 and 1 <= s2 <= 20):
        raise ValueError('val min = 1, val max = 20')

    if img.dtype != np.uint8:
        img = img2uint8(img)

    out = np.zeros(img.shape)

    if len(img.shape) > 2:
        # RGB image
        for d in range(img.shape[2]):
            vmin = searchvmin(cumhist(img[:, :, d]), img.shape[0], img.shape[1], s1)
            vmax = searchvmax(cumhist(img[:, :, d]), img.shape[0], img.shape[1], s2)
            out[:, :, d] = saturate_rescale_pixels(img[:, :, d], vmin, vmax)
    else:
        # Grayscale image
        vmin = searchvmin(cumhist(img), img.shape[0], img.shape[1], s1)
        vmax = searchvmax(cumhist(img), img.shape[0], img.shape[1], s2)
        out = saturate_rescale_pixels(img, vmin, vmax)

    return np.uint8(out)


if __name__ == '__main__':
    img = mpimg.imread('./images/lenna.png')

    s1 = 1.5
    s2 = 1.5

    print('Please Wait...')
    out = scb(img, s1, s2)
    plotresult(img, out, s1, s2)
    plt.show()
