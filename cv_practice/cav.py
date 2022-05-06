
import cv2 as cv
import numpy as np

# tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

h_size = 240
w_size = 320
hiimg = np.arange(h_size,dtype='float32').reshape((1,h_size))
wiimg = np.arange(w_size,dtype='float32').reshape((1,w_size))
#print(f"hiimg = {hiimg}\n")
#print(f"wiimg = {wiimg}\n")

ho_size = 1536
wo_size = 2304

hy = cv.resize(hiimg, (ho_size,1))
wy = cv.resize(wiimg, (wo_size,1))
#print(f"opencv resize hy = {hy}\n")
#print(f"opencv resize wy = {wy}\n")

htimg = tf.constant(hiimg[np.newaxis,:,:,np.newaxis])
wtimg = tf.constant(wiimg[np.newaxis,:,:,np.newaxis])
hoimg = tf.image.resize(htimg, [1,ho_size], method='bilinear').numpy()[0,:,:,0]
woimg = tf.image.resize(wtimg, [1,wo_size], method='bilinear').numpy()[0,:,:,0]
#print(f"tensorflow resize hoimg = {hoimg}\n")
#print(f"tensorflow resize woimg = {woimg}\n")

print(f"hy - hoimg = {hy - hoimg}\n")
print(f"wy - woimg = {wy - woimg}\n")

H = np.concatenate((np.floor(hoimg),hoimg, hoimg-np.floor(hoimg),np.ceil(hoimg)), axis=0).T
W = np.concatenate((np.floor(woimg),woimg, woimg-np.floor(woimg),np.ceil(woimg)), axis=0).T

Hcv = np.concatenate((np.floor(hy),hy, hy-np.floor(hy),np.ceil(hy)), axis=0).T
Wcv = np.concatenate((np.floor(wy),wy, wy-np.floor(wy),np.ceil(wy)), axis=0).T

with open('H.txt', 'w+') as hfp:
    np.savetxt(hfp, H, fmt='%20.16f')

with open('W.txt', 'w+') as wfp:
    np.savetxt(wfp, W, fmt='%20.16f')

with open('Hcv.txt', 'w+') as hfp:
    np.savetxt(hfp, Hcv, fmt='%20.16f')

with open('Wcv.txt', 'w+') as wfp:
    np.savetxt(wfp, Wcv, fmt='%20.16f')


with open('T.txt', 'w+') as tfp:
    np.savetxt(tfp, (wy-np.floor(wy)).T, fmt='%20.16f')
    
