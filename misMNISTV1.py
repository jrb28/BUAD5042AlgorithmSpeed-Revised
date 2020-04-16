# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:39:31 2020

@author: Jim
"""

import numpy as np
import matplotlib.pyplot as plt
import time

''' Load MNIST data set '''
print('Loading MNIST data...')
train_labels = np.genfromtxt('train_labels.gz',delimiter=',',dtype='int')
train_images = np.genfromtxt('train_images.gz',delimiter=',')

'''CONDITION DATA '''
''' Normalize range of data to [0,1] and reshape images '''
train_images = train_images.astype('float32') / 255
train_images = train_images.reshape((int(train_images.shape[0]/(28 * 28)), 28 * 28))
print('MNIST Data Loaded.')


''' The following block of code loads reference images for each of the 9 characters in a list of lists of lists.
    ref is a 3D list, ref[i][j][k] where
        i is in the range of 0 to 9 and refers to the integer labels for the reference images
        j is the index of the images with label i
        k is the index of the pixel in the image with label i, and image index j '''

''' Don't change the block of code below'''
print('Loading Reference Image Data...')
num_correct = 0
num_labels = 10  # number of labels 0-9
ref = []
for i in range(num_labels):
    f = open('./data/ref'+str(i)+'.csv','r')
    data = f.readlines()
    f.close()
    for i in range(len(data)):
        data[i] = data[i].split(',')
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])/255.0
    ref.append(data)
print('Reference Image Data Loaded...')

''' Set start time variable '''
start_time = time.time()

''' =================  YOU MAY CHANGE CODE BELOW THIS LINE =========================== '''

''' This variable, num_imgs, can be changed '''
num_imgs = 10   # number of images to check
num_imgs = min(num_imgs, train_images.shape[0])  # limit images ot the number avaialble

''' This block of code computes the sum of squared differences between a MNIST image and 
    each reference image '''    
imgs_dist = []     # list for accumulating  distances for all images
for i in range(num_imgs):   #train_images[:1]:
    img_dist = []  # list for accumulating  distances for different labels for an image
    for lab in ref:
        lab_dist = []  # list for accumulating distances for an image and each reference image for each label
        for img_ref in lab:
            dist = 0
            for p in range(len(train_images[i])):
                dist += (train_images[i][p] - img_ref[p])**2
            lab_dist.append(dist)
        img_dist.append(lab_dist)
    imgs_dist.append(img_dist)
        
''' Insert code here to use the sum of squared differences distances computed above to identify possibly mislabeled images '''
''' It is okay to change the code in the SSD computation block above in order to facilitate this task '''
''' The code below provides a sample of how the SSD disstances might be used to evaluate whether labels are correct '''
imgs_mislabeled = []  # put indeices
for i in range(len(imgs_dist)):
    score_min = []
    for lab in imgs_dist[i]:
        score_min.append(min(lab))
    print('Index:', i, 'Label:',train_labels[i], '    Min score label:', np.argmin(score_min), '    ', train_labels[i]==np.argmin(score_min), abs(score_min[np.argmin(score_min)] - score_min[train_labels[i]]))
    if train_labels[i]==np.argmin(score_min):
        num_correct += 1
    else: 
        imgs_mislabeled.append(i)
    ''' or, use this elif to append image labels to the imgs_mislabeled list under specified conditions '''
    #elif abs(score_min[np.argmin(score_min)] - score_min[train_labels[i]]) > 10.0:
    #    imgs_mislabeled.append(i)
    
'''  =================  YOU MAY CHANGE CODE ABOVE THIS LINE =========================== '''
    
''' compute execution time '''
print('Number of images analyzed:', num_imgs)
print('Execution time:', time.time() - start_time)
print('Average time per image:', (time.time() - start_time)/float(num_imgs))
print(str(num_imgs) + ' images analyzed.  ' + str(num_correct) + ' images labeled consistently with SSD.  ' + str(num_imgs - num_correct) + ' images possibly mislabeled.\n\n')

''' Code to view potentially mislabeled images whose indices are in the imgs_mislabeled list '''
for i in imgs_mislabeled:
    print('Index',i, '  Train label:', train_labels[i])
    fig,ax = plt.subplots()
    ax.imshow(train_images[i].reshape(28,28),cmap='gray')
    plt.show() 