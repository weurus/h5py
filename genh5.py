import h5py
import os
import cv2
import numpy as np

rgb = cv2.imread('0.png')
depth = np.load('0.npy')

h5f = h5py.File('gen.h5', 'w')

h5f.create_dataset('rgb', data=rgb)
h5f.create_dataset('depth', data=depth)

h5f.close()


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    rgb = f['rgb'][:].transpose(1,2,0)
    depth = f['depth'][:]
    return (rgb, depth)


data = load_h5('gen.h5')
print(data[0].shape, data[1].shape)
