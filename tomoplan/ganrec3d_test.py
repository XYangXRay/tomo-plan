import numpy as np
import tomopy 
import time
import matplotlib.pyplot as plt
import tifffile
from gan3d_rec import GANrec3d

data = tomopy.shepp3d()
ang = tomopy.angles(181)
prj = tomopy.project(data, ang, pad=False)

# plt.imshow(train_input)
# plt.show()

prj= prj/prj.max()

# print(data.min(), data.max())
start = time.time()

gan3d_object = GANrec3d(prj, ang, iter_num=5000)
rec = gan3d_object.recon
print(f'Reconstruction time is {time.time()-start} s.')

tifffile.imwrite('/nsls2/users/xyang4/data/recon_test_20220919.tiff', rec)