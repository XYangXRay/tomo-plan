from cgi import test
import numpy as np
from tensorflow.keras.models import load_model
import tomopy 
import matplotlib.pyplot as plt
from tomoplan.gan3d import GAN3d
import tifffile
import time

def nor_tomo(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())
    return img


data = tomopy.shepp3d()
ang = tomopy.angles(181)
prj = tomopy.project(data, ang, pad=False)
train_input = prj[0]


prj_test = tifffile.imread('/data/circle_prj.tiff')
test_input = prj_test[0]
test_input = test_input/test_input.max()

train_input = train_input/train_input.max()
# plt.imshow(train_input)
# plt.show()
train_output = np.swapaxes(data, 0, 1)
# plt.imshow(train_output[64,:,:])
# plt.show()
start = time.time()
train_obj = GAN3d(train_input, train_output, test_input, iter_num=1000)
# recon = train_obj.train
end = time.time()
# recon = recon.numpy()
# print('Running time is {}'.format(end - start))
# tifffile.imwrite('/data/3d_train.tiff', recon.reshape((128, 128, 128)))

# recon_model = load_model('/data/weights/3d_generator.h5')

# test_output = train_obj.predict
model = load_model('/data/weights/3d_generator.h5')
model.summary()
test_input = np.reshape(test_input, (1, 128, 128, 1))
test_output = model(test_input)
test_output = test_output.numpy()
# test_output = recon_model.predict(test_input)
tifffile.imwrite('/data/3d_predict.tiff', test_output.reshape((128, 128, 128)))