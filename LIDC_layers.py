import caffe

import os
import numpy as np
from PIL import Image

import random

class LIDCSegDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.lidc_dir = params['lidc_dir']
        #Train or val
        self.split = params['split']
        #This mean value needs an extra script
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        path_to_images = '/home/zhipeng/Work/Data/LIDC-annotation/Images'
        path_to_gts = '/home/zhipeng/Work/Data/LIDC-annotation/Label_1'
        self.indices_images = os.listdir(path_to_images)
        self.indices_gts = os.listdir(path_to_gts)
        self.idx = 1

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(1, len(self.indices_images))

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.idx)
        self.label = self.load_label(self.idx)
        # reshape tops to fit (leading 1 is for batch dimension)
        #What is the meaning of star
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(1, len(self.indices_images))
        else:
            self.idx += 1
            #if self.idx == len(self.indices_images):
            #    self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/Images/{}.tif'.format(self.lidc_dir, idx))
        in_ = np.array(im, dtype=np.uint16)
	max_val = np.amax(in_) + 0.0
	in_ = (in_ / max_val) * 255.
        #in_ = in_[:,:,::-1]
        in_ -= (self.mean / max_val) * 255.
	in_ = np.array(in_, dtype=np.float32)
        in_ = in_[np.newaxis, ...]
        value = in_
        in_ = np.insert(in_, 0, values = value, axis = 0)
        in_ = np.insert(in_, 0, values = value, axis = 0)
        #in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/Label_1/GT_{}.tif'.format(self.lidc_dir, idx))
        label = np.array(im, dtype=np.uint8)
	# value = label
	# label = label[np.newaxis, ...]
	# label = np.insert(label, 0, values = value, axis = 0)
	# label = np.insert(label, 0, values = value, axis = 0)
	# label = label.transpose((1,2,0))
	new_label = np.zeros(label.shape)
	for i in range(0, label.shape[0]):
		for j in range(0, label.shape[1]):
			if label[i][j] == 255:
				new_label[i][j] = 1
        new_label = new_label[np.newaxis, ...]
        return new_label


class LIDCValDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.lidc_dir = params['lidc_dir']
        #Train or val
        self.split = params['split']
        #This mean value needs an extra script
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        path_to_images = '/home/zhipeng/Work/Data/LIDC-annotation-val/Images'
        path_to_gts = '/home/zhipeng/Work/Data/LIDC-annotation-val/Label_1'
        self.indices_images = os.listdir(path_to_images)
        self.indices_gts = os.listdir(path_to_gts)
        self.idx = 1

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(1, len(self.indices_images))

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.idx)
        self.label = self.load_label(self.idx)
        # reshape tops to fit (leading 1 is for batch dimension)
        #What is the meaning of star
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(1, len(self.indices_images))
        else:
            self.idx += 1
            if self.idx == len(self.indices_images):
                self.idx = 1


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/Images/{}.tif'.format(self.lidc_dir, idx+3499))
        in_ = np.array(im, dtype=np.uint16)
	max_val = np.amax(in_) + 0.0
	in_ = (in_ / max_val) * 255.
        in_ -= (self.mean / max_val) * 255.
        in_ = np.array(in_, dtype=np.float32)
 	in_ = in_[np.newaxis, ...]
        value = in_
        in_ = np.insert(in_, 0, values = value, axis = 0)
        in_ = np.insert(in_, 0, values = value, axis = 0)
        #in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/Label_1/GT_{}.tif'.format(self.lidc_dir, idx+3499))
        label = np.array(im, dtype=np.uint8)
	# value = label
	# label = label[np.newaxis, ...]
	# label = np.insert(label, 0, values = value, axis = 0)
	# label = np.insert(label, 0, values = value, axis = 0)
	# label = label.transpose((1,2,0))
	new_label = np.zeros(label.shape)
	for i in range(0, label.shape[0]):
		for j in range(0, label.shape[1]):
			if label[i][j] == 255:
				new_label[i][j] = 1
        new_label = new_label[np.newaxis, ...]
        return new_label



'''
class VOCSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/SegmentationClass/{}.png'.format(self.voc_dir, idx))
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label


class SBDDSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.sbdd_dir = params['sbdd_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.sbdd_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/img/{}.jpg'.format(self.sbdd_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label
'''
