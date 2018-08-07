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
    train_indices = indices[:int(train_prop * len(imlist))]
    val_indices = indices[int(train_prop) * len(imlist):]
    for i in train_indices:
        im = imlist[i]
        sh.copy(os.path.join(path, im), os.path.join(newpath_train, im))
    for i in val_indices:
        im = imlist[i]
        sh.copy(os.path.join(path, im), os.path.join(newpath_val, im))



