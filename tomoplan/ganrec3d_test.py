import numpy as np
import tomopy 
import matplotlib.pyplot as plt
import tifffile
from gan3d_rec import GANrec3d

data = tomopy.shepp3d()
ang = tomopy.angles(181)
prj = tomopy.project(data, ang, pad=False)
train_input = prj[0]
# plt.imshow(train_input)
# plt.show()
print(train_input.min(), train_input.max())
print(data.min(), data.max())

gan3d_object = GANrec3d(prj, ang, iter_num=400)
rec = gan3d_object.recon

tifffile.imwrite('/Users/xiaogangyang/data/recon_test.tiff', rec)