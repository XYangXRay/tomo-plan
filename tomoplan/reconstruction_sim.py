import tomopy
from tomoplan.gan3d import GAN3d
import numpy as np
import matplotlib.pyplot as plt
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
train_input = np.zeros((100, 128, 128))
train_output = np.zeros((100, 128, 128, 128))

for i in range(100):
    ncl = np.random.randint(2, 20)
    prj, obj = obj_generate(dx, dy, dz, 10, 4, 20)
    train_input[i] = prj[0]
    train_output[i] = obj

train_obj = GAN3d(train_input, train_output, iter_num=1000)
train_obj.train
# tifffile.imwrite('/data/3d_tomo/prj_test.tiff', prj)
# tifffile.imwrite('/data/3d_tomo/obj_test.tiff', obj)
