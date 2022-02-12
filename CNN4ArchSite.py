# %%
# read band- and mask-images. All in tiff format
from distutils.sysconfig import customize_compiler
import cv2
import os
from matplotlib import pyplot as plt
import tifffile
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features

ProfCurv = tifffile.imread(".\\InData\\UAV_final_bands\\B6_ProfCurv.tif")
NDVI = tifffile.imread(".\\InData\\UAV_final_bands\\B3_NDVI.tif")
DEV = tifffile.imread(".\\InData\\UAV_final_bands\\B5_DEV.tif")

# only use 3 images: ProfCurv, NDVI, and DEV for now
# because the default Unet and Resnet allows only three channels (as most images with RGB).
# can change to 6 channels but need work

# adjust values in each image to greater than 0
# set no data to 0
# check max and min values in each image
# scale the values to a range 0-1 for each image

# ProfCurv.tif image has values -505.892 to 458.889. 
# Background cell value: 3.4e+38 

ProfCurv_revised = ProfCurv + 510
ProfCurv_revised[ProfCurv_revised > 970] = 0
ProfCurv = ProfCurv_revised/ProfCurv_revised.max()
print(f"ProfCurv: min:{ProfCurv.min()}, max:{ProfCurv.max()}, std:{ProfCurv.std():0.2f},the number of background cells = {(ProfCurv == 0).sum()}")

# NDVI values range from -1 to 1. 
# Background cell value: -3.4e+38
NDVI_revised = NDVI + 1.5
NDVI_revised[NDVI_revised < 0] = 0
NDVI = NDVI_revised/NDVI_revised.max()
print(f" NDVI: min:{NDVI.min()}, max:{NDVI.max()}, std:{NDVI.std():0.2f}, the number of background cells: {(NDVI == 0).sum()}")

# DEV values range from -3.86105 to 6.20381
# Background -3.4e+38 

DEV_revised = DEV + 5
DEV_revised[DEV_revised < 0] = 0
DEV = DEV_revised/DEV_revised.max()
print(f" DEV: min:{DEV.min()}, max:{DEV.max()}, std:{DEV.std():0.2f}, the number of background cells: {(DEV == 0).sum()}")
#
# merge all input images to create a composite image with 3 channels
# NDVI, DEV, and ProfCurv have the same min and max values
comp_img = cv2.merge((NDVI, DEV, ProfCurv))
plt.imshow(comp_img)
plt.show()

#
# divide the mask file (a grid with categories of archaeological features: 1-7,
# plus background = 0) into 256 x 256 image patches.
# 

mask_file = r'./InData/mask.tif'
mask = cv2.imread(mask_file)
# mask is a numpy array (4299, 3956, 3).
# only burn values to band 1 which is red, but cv reads the red band the last,
# so the red band corresponds to channel #2
mask = mask[:,:,2]
plt.imshow(mask, vmin=0, vmax=7, cmap='jet')
plt.colorbar()
plt.show()

# merge input images into a composite image of (4299, 3956, 8)
# use patchify and unpatchify. However, unpatchify requires (width-patch) mod step size = 0
# so we will use the following:
# composite image size = (4299, 3956, 6)
# patch size = (256, 256)
# step size = 256 (if overlaps, will have prediction conflicts when unpatchify)

# have problems with the Unet model.
# change dimensions to 3840 x 3840
# change 4299 to 3840 by removing 230 at the bottom and 229 on the top
# change 3956 to 3840 by removing 58 on the left and 58 on the right

p_w = 256
p_h = 256
p_c = 3
p_s = 256

comp_img_trimmed = comp_img[230:4070, 58:3898, :]
mask_trimmed = mask[230:4070, 58:3898]

from patchify import patchify, unpatchify
img_patches = patchify(comp_img_trimmed, (p_w, p_h, p_c), step=p_s)
assert img_patches.shape == (15, 15, 1, 256, 256, 3)
mask_patches = patchify(mask_trimmed, (p_w, p_h), step=p_s)

# the non-overlapping image patches will
# be used for final prediction using the trained model
patched_img_prediction = img_patches.reshape(-1, p_w, p_h, p_c)
patched_mask_prediction = mask_patches.reshape(-1, p_w, p_h, 1)

# create image patches as labelled data for model training and testing.
# okay to have overlaps since we are not going to unpatchify the training patches
# will use the trimmed comp_img since the removed pixels are background pixels

# set step = 64 for 75% overlaps
patched_img_modeling = patchify(comp_img_trimmed, (p_w, p_h, p_c), step=64)
patched_mask_modeling = patchify(mask_trimmed, (p_w, p_h), step=64)
# simplify the names
patched_img = patched_img_modeling.reshape(-1, 256, 256, 3)
patched_mask = patched_mask_modeling.reshape(-1, 256, 256)

print("Done with creating mask patches and image patches")


#
# prepare training, validation, and testing imagery patches

# encode multiple categories for 7 classes.
# need to flatten the ndarray first

from sklearn.preprocessing import LabelEncoder
import numpy as np

labelencoder = LabelEncoder()
n, h, w = patched_mask.shape
patched_mask_flattened = patched_mask.reshape(-1)
patched_mask_encoded = labelencoder.fit_transform(patched_mask_flattened)
patched_mask_encoded = patched_mask_encoded.reshape(n, h, w, 1)
np.unique(patched_mask_encoded) # assert category values from 0-7


# The codes below built upon tensorflow, which requires GPU processors. The study uses Ndivia GPUs.
# Each manufacture has instructions on how to set up their GPU. Ndivia instructions are available at
# https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

os.environ['SM_FRAMEWORK'] = 'tf.keras'
import tensorflow as tf
from keras import optimizer_v2
import segmentation_models as sm
from keras.utils.np_utils import to_categorical
mask_cat = to_categorical(patched_mask_encoded, num_classes=8)

from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(patched_img, mask_cat,
                                        test_size=0.20, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X1, y1,
                                        test_size=0.25, random_state=2)

print("Finish dividing image patches to training, validation, and testing")


n_class = 8
activation_func = 'softmax'
LR = 0.001

# tried Adam, Adamax, and SGD
# for Adam and Adamax: use dice and categoical focal losses
# for SGD: use dice loss only because SGD decents slowly and does not fully converges at 1000 epochs,
#          and categorical focal losses are very small after 20 epochs. 

optim = tf.keras.optimizers.Adamax(LR)

unique, counts = np.unique(patched_mask_encoded, return_counts=True)
frequencies = np.asarray((unique, counts)).T
freqDF = pd.DataFrame(frequencies, columns=['cat', 'freq'])
freqDF['rawweight'] = freqDF['freq'].sum()/(n_class * freqDF['freq'])
freqDF['weight'] = freqDF['rawweight']/freqDF['rawweight'].sum()


weights = np.array(freqDF['weight']).round(3)

# check weights:
# array([0.001 , 0.02, 0.044, 0.3, 0.0156, 0.044, 0.557, 0.019])


# dice loss is based on dice coefficient (mostly F1 score) to balance precision and recall for all categories
# categorical focal loss down-weighs well classified examples and thus more on the "not" cases.
# In this study, categorical focal loss has much smaller values than dice loss

dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + 10*focal_loss

# have to have 50% or more overlaps to consider detecting an object
metrics_use = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# add callbacks and early stopping
# need to normalize the image channels

BACKBONE_res = 'resnet34'
preprocess_input_res = sm.get_preprocessing(BACKBONE_res)

X_train_res = preprocess_input_res(X_train)
X_val_res = preprocess_input_res(X_val)

model_res = sm.Unet(BACKBONE_res, encoder_weights='imagenet', classes=n_class, activation=activation_func)
model_res.compile(optim, total_loss, metrics=metrics_use)
print(model_res.summary())

# add callbacks and early stops if the model runs too long

history_res = model_res.fit(X_train_res, y_train, epochs=100, batch_size=32,
                            verbose=True, validation_data=(X_val_res, y_val),
                            use_multiprocessing=True)

# save the model. The following line saved the proposed model reported in the paper.
model_res.save('./saved_models/AdamaxRes34_backbone_100epochs.hdf5')

print("Finish model development")


# plot the training/validation accuracy and loss at each epoch
loss = history_res.history['loss']
val_loss = history_res.history['val_loss']
iou = history_res.history['iou_score']
val_iou = history_res.history['val_iou_score']
epochs = range(1, len(loss)+1)

fig, axes = plt.subplots(2, figsize=(5, 7))
axes[0].plot(epochs, loss, 'y', label='Training Loss')
axes[0].plot(epochs, val_loss, 'r', label='Validation Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[1].plot(epochs, iou, 'y', label='Training Accuracy')
axes[1].plot(epochs, val_iou, 'r', label='Validation Accuracy')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy (IOU)')
axes[1].legend()
plt.savefig('./OutData/AdamaxLossAccu.jpeg', dpi=300)



# apply the model to testing data
# plot out the test and pred images (run the codes repeatly to inspect
# multiple test image patches)

import random
test_image_number = random.randint(0, len(X_test))
test_image = X_test[test_image_number]
label_image = y_test[test_image_number]
# the first dim should be the number of images
test_image_input = np.expand_dims(test_image, 0)
# preprocess the input for the resnet model
test_image_input_res = preprocess_input_res(test_image_input)
test_image_pred = model_res.predict(test_image_input_res)
test_image_pred_cat = np.argmax(test_image_pred, axis=3)[0,:,:]

from keras.metrics import MeanIoU
IOU_pred = MeanIoU(num_classes=n_class)
IOU_pred.update_state(test_image_input_res[:,:,:,0], test_image_pred_cat)
print("IOU = ", IOU_pred.result().numpy())
# the prediction IOU varies on individual test images. 

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_image[:,:,0], cmap='jet')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(label_image[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_image_pred_cat, cmap='jet')
plt.savefig('./OutData/AdamaxPredicted.jpeg', dpi=300)

# Applied the model to all image patches and mosaicked the prediction 
# to construct the entire predicted map

patched_prediction = model_res.predict(patched_img_prediction)
predicted_classes = np.argmax(patched_prediction, axis=3)
IOU_pred = MeanIoU(num_classes=n_class)
IOU_pred.update_state(patched_mask_prediction[:,:,:,0], predicted_classes)
print("Mean IOU = ", IOU_pred.result().numpy())

# need to reverse what has done to the original image to create patches
# -- change 4299 to 3840 by removing 230 at the bottom and 229 on the top
# -- change 3956 to 3840 by removing 58 on the left and 58 on the right
# -- pad zeros to these removed cells


predicted_classes = predicted_classes.reshape(15, 15, 256, 256)
predicted_classes = unpatchify(predicted_classes, imsize=(3840, 3840))
np.unique(predicted_classes)  # make sure still have 8 classes 0 to 7

predicted_classes_padded = np.zeros((4299, 3956))
predicted_classes_padded[:predicted_classes.shape[0], :predicted_classes.shape[1]] = predicted_classes

plt.figure(figsize=(5,9))
plt.imshow(predicted_classes, cmap='jet')
plt.savefig('./OutData/AdamaxPredictionMap.jpeg', dpi=300)



# %%
