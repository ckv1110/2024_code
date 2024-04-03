import tifffile as tiff
import glob
from modules.features import get_deconvolved_stains, get_nuclei_features, get_centroids
import os
import multiprocessing

exp = 'T164'
im_path = glob.glob(f'./example_img/*{exp}*HnE*.tiff')[0]
im = tiff.TiffReader(im_path).asarray()
nuc_seg_path = glob.glob(f'./example_img/*{exp}*prediction*.tif')[0]
nuclei_seg_im = tiff.TiffReader(nuc_seg_path).asarray()

stain_colour_map = {'hematoxylin': [0.52, 0.74, 0.42],
                    'eosin':       [0.27, 0.71, 0.65],
                    'dab':         [0.27, 0.57, 0.78],
                    'null':        [0.00, 0.00, 0.00],
                    }

nuc_stain_im, _ = get_deconvolved_stains(im, stain_colour_map)
num_processes = multiprocessing.cpu_count()
# print("Number of CPU cores:", num_processes)
# print("Using", num_processes - 4, "processes for feature extraction.")
if __name__ == '__main__':
    centroids = get_centroids(nuclei_seg_im)
    print(centroids)
    # centroids.to_csv(f'{os.getcwd()}/example_img/{exp}_feature_set/{exp}_centroids.csv', index=False)

    # get_nuclei_features(nuc_stain_im, nuclei_seg_im, save_path=f'{os.getcwd()}/example_img/{exp}_feature_set', file_name=exp, verbose=True)