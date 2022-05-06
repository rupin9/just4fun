

import cv2 as cv
import numpy as np

x = np.array([[0, 1, 2, 3, 4, 5]], dtype='float32')
print(f"x = {x}\n")

# opencv resize
y = cv.resize(x, (12,1)).tolist()
print(f"opencv resize y = {y}\n")

# Affine
M = np.array([[2, 0, 0], [0, 1, 0]], dtype='float32')
z = cv.warpAffine(x, M, dsize=(12,1), borderMode=cv.BORDER_REFLECT).tolist()
print(f"warpAffine z = {z}\n")

# python (shift 0.5) and (shift -0.5)
w = lambda i, s: (i+0.5)*(1./s)-0.5  # i is the target index and s is the scale
print(f"python shift(0.5 -> -0.5) = {[w(i, 2) for i in range(12)]}\n")

# tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

t = tf.constant(x[np.newaxis,:,:,np.newaxis])   # tensorflow needs a batch axis and a channel axis
#sess = tf.Session()
#print(sess.run(tf.image.resize_bilinear(t, [1,12]))[0,:,:,0].tolist())

v = tf.compat.v1.image.resize_bilinear(t, [1,12]).numpy()[0,:,:,0].tolist()
print(f"tensorflow resize v = {v}\n")

u = tf.compat.v1.image.resize_bilinear(t, [1,12], align_corners=True).numpy()[0,:,:,0].tolist()
print(f"tensorflow resize u= {u}\n")

q = tf.image.resize(t, [1,12], method='bilinear').numpy()[0,:,:,0].tolist()
print(f"tensorflow resize q = {q}\n")

################################################################################

iimg = np.arange(5,dtype='float32').reshape((1,5))
print(f"iimg size = {iimg} {iimg.size}\n")

i_size = iimg.size
scale = 2;
timg = tf.constant(iimg[np.newaxis,:,:,np.newaxis])
oimg = tf.image.resize(timg, [1,scale*i_size], method='bilinear').numpy()[0,:,:,0]
print(f"tensorflow resize oimg size = {oimg} {oimg.size}\n")

