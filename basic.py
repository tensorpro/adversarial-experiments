from __future__ import print_function
## Setup

import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

## GG

# get source image and label
image, label = foolbox.utils.imagenet_example()
image = image[:,:,::-1]
print('original', kmodel.predict(np.array([image])).argmax())
# apply attack on source image
attack = foolbox.attacks.GradientAttack(fmodel)
adversarial = attack(image, label)
print('adversarial', kmodel.predict(np.array([adversarial])).argmax())

## Add noise
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# ia.seed(1)

seq = iaa.Sequential([
    # iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-15, 15),
        shear=(-8, 8)
    )
]) # apply augmenters in random order

tst = 1000

raw_aug = seq.augment_images([image]*tst)
ad_aug = seq.augment_images([adversarial]*tst)

## Test

raw_p = [kmodel.predict(np.expand_dims(x,0)).argmax() for x in raw_aug]
ad_p = [kmodel.predict(np.expand_dims(x,0)).argmax() for x in ad_aug]

## Eval

raw_acc = [kmodel.predict(np.expand_dims(x,0)).argmax() for x in raw_aug]
ad_acc = [kmodel.predict(np.expand_dims(x,0)).argmax() for x in ad_aug]

print('raw acc', raw_acc)
print('ad acc', raw_acc)

## Vis

show = False

if show:
    accuracy_scaled = print(np.mean(raw_p==ad_p))
    images_aug = seq.augment_images([image]*6)
    images_aug = seq.augment_images([adversarial]*6)
    # plt.imshow(images_aug[0][:,:,::-1]/255.);plt.show()


    for img in images_aug:
        plt.title(str(kmodel.predict(np.array([img])).argmax()))
        plt.imshow(img[:,:,::-1]);plt.show()
        # plt.title(str(kmodel.predict(np.array([img])).argmax()))
        # plt.imshow(img[:,:,::-1]);plt.show()


