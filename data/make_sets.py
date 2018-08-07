import os
import numpy as np
import shutil as sh

trainpath = 'data/train/'
testpath = 'data/test'
newtrain = "data/competition/train"
newval = "data/competition/val"
newtest = "data/competition/test"

os.makedirs(newtrain, exist_ok=True)
os.makedirs(newval, exist_ok=True)
os.makedirs(newtest, exist_ok=True)

train_folders = os.listdir(trainpath)
test_folders = os.listdir(testpath)

train_prop = 0.8

for tr in train_folders:
    path = os.path.join(trainpath, tr)
    newpath_train = os.path.join(newtrain, tr)
    os.makedirs(newpath_train, exist_ok=True)
    newpath_val = os.path.join(newval, tr)
    os.makedirs(newpath_val, exist_ok=True)
    imlist = os.listdir(path)
    indices = np.random.permutation(len(imlist))
    ntrain = int(train_prop * len(imlist))
    train_indices = indices[:ntrain]
    val_indices = indices[ntrain:]
    assert(len(val_indices) < len(train_indices))
    for i in train_indices:
        im = imlist[i]
        sh.copy(os.path.join(path, im), os.path.join(newpath_train, im))
    for i in val_indices:
        im = imlist[i]
        sh.copy(os.path.join(path, im), os.path.join(newpath_val, im))

os.makedirs(os.path.join(newtest, 'all'), exist_ok=True)

for te in test_folders:
    path = os.path.join(testpath, te)
    ims = os.listdir(path)
    for im in ims:
        impath = os.path.join(path, im)
        sh.copy(impath, os.path.join(newtest, "all", im))




