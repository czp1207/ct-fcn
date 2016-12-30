from PIL import Image
import numpy as np
import os
 
Images_path = '/media/zhipeng/Windows7_OS/LIDC-annotation/Images'
image_ls = os.listdir(Images_path)
index = 1
mean = 0;
for i in range(0, len(image_ls)):
	im = Image.open('{}/{}.tif'.format(Images_path, index))
	in_ = np.array(im)
	im_mean = np.mean(in_)
	mean += im_mean

mean = mean/len(image_ls)

print mean