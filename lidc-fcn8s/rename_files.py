import os

file_list = os.listdir('/media/zhipeng/Windows7_OS/LIDC-annotation')
for i in range(0,len(file_list)):
	file_list[i] = '/media/zhipeng/Windows7_OS/LIDC-annotation/'+file_list[i]

data_list = []
for i in range(0,len(file_list)):
	data_list = os.listdir(file_list[i])
	for j in range(0, len(data_list)):
		new_name = 'set_{}'.format(j+1)
		src = '{}/{}'.format(file_list[i], data_list[j])
		dst = '{}/{}'.format(file_list[i], new_name)
		#os.rename(src, dst)

image_list = []
for i in range(0,len(file_list)):
	#print file_list[i]
	if 'images' in file_list[i]:
		data_list = os.listdir(file_list[i])
		for j in range(0, len(data_list)):
			path_to_image = file_list[i] + '/' + data_list[j]
			image_list = os.listdir(path_to_image)
			index = 1
			for k in range(0, len(image_list)):
				if '.tif' in image_list[k]:
					new_image_name = '{}/{}_{}.tif'.format(path_to_image, data_list[j], index)
					index = index + 1
					src = '{}/{}'.format(path_to_image, image_list[k])
					#os.rename(src, new_image_name)
	if 'gts' in file_list[i]:
		data_list = os.listdir(file_list[i])
		for j in range(0, len(data_list)):
			path_to_image= file_list[i] + '/' + data_list[j]
			image_list = os.listdir(path_to_image)
			index = 1
			for k in range(0, len(image_list)):
				if '.txt' not in image_list[k]:
					path_to_gts= file_list[i] + '/' + data_list[j] + '/' + image_list[k]
					gts_list = os.listdir(path_to_gts)
					for l in range(0, len(gts_list)):
						new_gts_name = '{}/{}_{}_GT_id_{}.tif'.format(path_to_gts, data_list[j], index, l+1)
						src = '{}/{}'.format(path_to_gts, gts_list[l])
						os.rename(src, new_gts_name)
					index = index + 1
