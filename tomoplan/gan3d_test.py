import numpy as np
import tomopy 
import matplotlib.pyplot as plt
from tomoplan.gan3d import GAN3d
import tifffile

data = tomopy.shepp3d()
ang = tomopy.angles(181)
prj = tomopy.project(data, ang, pad=False)
train_input = prj[0]
train_input = train_input/train_input.max()
# plt.imshow(train_input)
# plt.show()
train_output = np.swapaxes(data, 0, 1)
# plt.imshow(train_output[64,:,:])
# plt.show()
train_obj = GAN3d(train_input, train_output, iter_num=5000)
recon = train_obj.train
recon = recon.numpy()

tifffile.imwrite('/data/3d_test.tiff', recon.reshape((128, 128, 128)))