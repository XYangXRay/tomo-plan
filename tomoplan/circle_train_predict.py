import tomopy
from tomoplan.gan3d import GAN3d
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
import tifffile

def circleval(px, py, pz, cx, cy, cz, ri):
    circle = (px - cx)**2 + (py - cy)**2 + (pz - cz)**2< ri**2
    return circle
def drawcircle(img, x, y, z, rad):


    dx, dy, dz = img.shape
    
    _x = np.arange(dx)
    _y = np.arange(dy)
    _z = np.arange(dz)
    
    px, py, pz = np.meshgrid(_x, _y, _z)
    
#    Parallel(n_jobs = 16)(delayed)
    for m in range(len(rad)):
        
        img += circleval(px, py, pz, x[m], y[m], z[m], rad[m])
        #img += (px - x[m])**2 + (py - y[m])**2 + (pz - z[m])**2< rad[m]**2

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
    
    #prj = np.swapaxes(prj, 0, 1)
    return prj, obj

def _round_to_even(num):
    return (np.ceil(num / 2.) * 2).astype('int')

dx, dy , dz = 128, 128, 128
train_input = np.zeros((20, 128, 128))
train_output = np.zeros((20, 128, 128, 128))

for i in range(20):
    ncl = np.random.randint(2, 20)
    prj, obj = obj_generate(dx, dy, dz, 10, 4, 20)
    train_input[i,:, :] = prj[0]
    train_output[i] = obj

train_input = train_input.astype('float32')
train_output = train_output.astype('float32')

train_obj = GAN3d(train_input, train_output, iter_num=1000, save_wpath='/Users/xiaogangyang/pyprojects/weights/')
train_obj.train

model = load_model('/Users/xiaogangyang/pyprojects/weights/3d_generator.h5')
model.summary()
test_input = np.reshape(train_input[10], (1, 128, 128, 1))
test_output = model(test_input)
test_output = test_output.numpy()
# test_output = recon_model.predict(test_input)
tifffile.imwrite('/Users/xiaogangyang/pyprojects/3d_predict.tiff', test_output.reshape((128, 128, 128)))