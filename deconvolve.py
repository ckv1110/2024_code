from modules import load_image, get_deconvolved_stains
import tifffile as tiff
import glob
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2hed, hed2rgb, separate_stains, combine_stains
import numpy as np
from scipy import linalg
import time
import gc

def deconvolve_scikit_p(im_input, stain_colour_map):
    # im_hed = rgb2hed(im_input)
    W = np.array([[0.49, 0.78, 0.37], [0.39, 0.71, 0.59], [0.80, -0.57, 0.16]])
    
    W_inv = linalg.inv(W)

    im_hed = separate_stains(im_input, W_inv)
    null = np.zeros_like(im_hed[:, :, 0])
    print('Performing color deconvolution...')
    im_h = combine_stains(np.stack((im_hed[:, :, 0], null, null), axis=-1), W)
    im_e = combine_stains(np.stack((null, im_hed[:, :, 1], null), axis=-1), W)
    im_d = combine_stains(np.stack((null, null, im_hed[:, :, 2]), axis=-1), W)

    # im_h = hed2rgb(np.stack((im_hed[:, :, 0], null, null), axis=-1))
    # im_e = hed2rgb(np.stack((null, im_hed[:, :, 1], null), axis=-1))
    # im_d = hed2rgb(np.stack((null, null, im_hed[:, :, 2]), axis=-1))
    print('Completed color deconvolution.')
    return im_h, im_e, im_d

if __name__ == '__main__':
    NAME = 'T19'
    path = os.getcwd()
    im_file = glob.glob(f'{path}/example_img/*{NAME}*HnE*.tiff')[0]
    print('Loading HnE image...')
    im = load_image(im_file, as_tf=False)
    print('Loading complete...')
    stain_matrix = {'hematoxylin': [0.52, 0.74, 0.42],
                    'eosin':       [0.27, 0.71, 0.65],
                    'dab':         [0.27, 0.57, 0.78],
                    'null':        [0.00, 0.00, 0.00],
                    }
    

    time1 = time.time()
    h, e, d = deconvolve_scikit_p(im, stain_matrix)
    time2 = time.time()
    print(f'Time taken for scikit deconvolution: {time2-time1}s')
    del h, e, d, time1, time2
    gc.collect()

    time1 = time.time()
    im_deconv = get_deconvolved_stains(im, stain_matrix)
    time2 = time.time()
    print(f'Time taken for htk deconvolution: {time2-time1}s')
    del im_deconv, time1, time2
    gc.collect()


    # print('Saving images...')
    # tiff.imwrite(f'./example_img/{NAME}_Hematoxylin.tif', h.astype('uint16'))
    # tiff.imwrite(f'./example_img/{NAME}_Eosin.tif', e.astype('uint16'))

    # Display
    # fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    # ax = axes.ravel()

    # ax[0].imshow(im)
    # ax[0].set_title("Original image")

    # ax[1].imshow(h)
    # ax[1].set_title("Hematoxylin")

    # ax[2].imshow(e)
    # ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image

    # ax[3].imshow(d)
    # ax[3].set_title("DAB")

    # for a in ax.ravel():
    #     a.axis('off')

    # fig.tight_layout()