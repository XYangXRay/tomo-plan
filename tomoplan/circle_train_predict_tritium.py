import tomopy
from tomoplan.gan3d import GAN3d
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
# from numba import njit, prange
import os
from tensorflow.keras.models import load_model
import tifffile

def circleval(px, py, pz, cx, cy, cz, ri):
    circle = (px - cx)**2 + (py - cy)**2 + (pz - cz)**2< ri**2
    return circle

# @njit(parallel=True)
def drawcircle(img, x, y, z, rad):


    dx, dy, dz = img.shape
    
    _x = np.arange(dx)
    _y = np.arange(dy)
    _z = np.arange(dz)
    
    px, py, pz = np.meshgrid(_x, _y, _z)
    
#    Parallel(n_jobs = 16)(delayed)
    for m in range(len(rad)):
        
        # img += circleval(px, py, pz, x[m], y[m], z[m], rad[m])
        img += (px - x[m])**2 + (py - y[m])**2 + (pz - z[m])**2< rad[m]**2

    return img

def obj_generate(dx, dy, dz, ncl, sml, lrg):
    # random circle generation
    # dx = 256  # size of x-axis
    # dy = 256  # size of y-axis
    # ncl = 10  # number of circles in an image frame
    # sml = 4  # smallest radius of the circle
    # lrg = 20  # largest radius of the circle
    # initial image frame
    x = np.random.randint(dx//4, dx//4*3, ncl)
    y = np.random.randint(dx//4, dx//4*3, ncl)
    z = np.random.randint(dx//4, dx//4*3, ncl)
      # print(x, y, z)
    # cen = _round_to_even(np.sqrt(dx **2 + dy **2) + 2)/2+cen_shift
   # cen = dx/2+cen_shift
    rad = np.random.randint(sml, lrg, ncl)
    obj = np.zeros((dx, dy, dz))
 #   for m in range(img_num):
    obj = drawcircle(obj, x, y, z, rad)
    # calculate the sinograms of obj
    ang = tomopy.angles(180, ang1=0.0, ang2=180.0)
    prj = tomopy.project(obj, ang, pad=False)  # <- this produces sinograms
    # change data range to 0-1
    prj = prj/prj.max()
    # swap the axis as the projection direction
    obj = obj/obj.max()
    obj = np.swapaxes(obj, 0, 1)
    
    #prj = np.swapaxes(prj, 0, 1)
    return prj, obj

def _round_to_even(num):
    return (np.ceil(num / 2.) * 2).astype('int')
  
batch_size = 10
data_size =100
dx, dy , dz = 128, 128, 128
train_input = np.zeros((data_size, 128, 128), dtype='float32')
train_output = np.zeros((data_size, 128, 128, 128), dtype='float32')

print('start to generate training data')
for i in range(data_size):
    ncl = np.random.randint(2, 20)
    prj, obj = obj_generate(dx, dy, dz, 10, 4, 20)
    train_input[i,:, :] = prj[0]
    train_output[i] = obj
    print('Generating data: {}'.format(i))
clear_output(wait=True)
# tifffile.imsave('/data/3d_tomo/train_circle/train_input.tiff', train_input)
# # tifffile.imwrite('/data/3d_tomo/train_circle/train_out.tiff', train_output, imagej = True)
# for i in range(len(train_output)):
#     fname = '/data/3d_tomo/train_circle/train_output/train_output' +  "-%03d" % (i) +'.tiff'
#     tifffile.imsave(fname, train_output[i])
   
def predict(fname_model, test_data, batch_size, fname_save):
    model = load_model(fname_model)
    test_input = np.zeros((batch_size,128,128,1))
    test_input[0] = test_data.reshape((128, 128,1))
    test_output = model(test_input)
    test_output = test_output.numpy()[0]
    tifffile.imwrite(fname_save, test_output.reshape((128, 128, 128)))

# train_input = tifffile.imread('/data/3d_tomo/train_circle/train_input.tiff')
# dirname = '/data/3d_tomo/train_circle/train_output/'
# train_output = []
# for fname in os.listdir(dirname):
#     print(fname)
#     data= tifffile.imread(os.path.join(dirname, fname))
#     # dataarray = np.array(data)
#     train_output.append(data)
# train_output = np.asarray(train_output)
# print(train_output.dtype, train_output.shape)


train_input = train_input[:data_size]
train_output = train_output[:data_size]
print('start to train the model')
train_obj = GAN3d(train_input, train_output, 
                  iter_num=2000, batch_size = batch_size, 
                  g_learning_rate = 5e-4, d_learning_rate = 1e-6, 
                #   init_wpath = '/nsls2/users/xyang4/data/',
                  save_wpath='/nsls2/users/xyang4/data/tritium/')
train_obj.train

print('training has been done')

data = tifffile.imread('/nsls2/users/xyang4/data/tritium/proj_Ni_K.tif')
test_data = np.zeros((128, 128))
test_data[20:101, 3:124] = data[0]

predict('/nsls2/users/xyang4/data/tritium/3d_generator.h5',
        test_data,
        batch_size,
        '/nsls2/users/xyang4/data/tritium/exp_predict_10_l10_20220902.tiff')

test_data = train_input[0]
predict('/nsls2/users/xyang4/data/tritium/3d_generator.h5',
        test_data,
        batch_size,
        '/nsls2/users/xyang4/data/tritium/test_predict_10_l10_20220902.tiff')


