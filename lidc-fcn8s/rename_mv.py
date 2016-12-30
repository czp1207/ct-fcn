import os
import shutil
import string

file_list = os.listdir('/media/zhipeng/Windows7_OS/LIDC-annotation')
for i in range(0,len(file_list)):
	file_list[i] = '/media/zhipeng/Windows7_OS/LIDC-annotation/'+file_list[i]
'''
data_list = []
for i in range(0,len(file_list)):
	data_list = os.listdir(file_list[i])
	for j in range(0, len(data_list)):
		new_name = 'set_{}'.format(j+1)
		src = '{}/{}'.format(file_list[i], data_list[j])
		dst = '{}/{}'.format(file_list[i], new_name)
		#os.rename(src, dst)
'''

image_list = []
for i in range(0,len(file_list)):
	#print file_list[i]
	index = 1
	index_2 = 1
	if 'images' in file_list[i]:
		images_dst = '/media/zhipeng/Windows7_OS/LIDC-annotation/Images'
		gts_dst = '/media/zhipeng/Windows7_OS/LIDC-annotation/Label_1'
		data_list = os.listdir(file_list[i])
		for j in range(0, len(data_list)):
			path_to_image = file_list[i] + '/' + data_list[j]
			image_list = os.listdir(path_to_image)
			for k in range(0, len(image_list)):
				if '.tif' in image_list[k]:
					src = '{}/{}'.format(path_to_image, image_list[k])
					gts_image = string.replace(image_list[k], '.tif', '')
					gts_src = '/media/zhipeng/Windows7_OS/LIDC-annotation/' + 'gts/' + data_list[j] + '/' + gts_image
					gts_ls = os.listdir(gts_src)
					for l in range(0, len(gts_ls)):
						if 'id1' in gts_ls[l]:
							path_to_gts = gts_src + '/' + gts_ls[l]
							shutil.copy(path_to_gts, gts_dst)
							new_gts_name = '{}/GT_{}.tif'.format(gts_dst, index)
							os.rename(gts_dst + '/' + gts_ls[l], new_gts_name)
							index_2 += 1
					shutil.copy(src, images_dst)
					new_src = images_dst + '/' + image_list[k]
					new_image_name = '{}/{}.tif'.format(images_dst, index)
					os.rename(new_src, new_image_name)
					index += 1

					if index == index_2: 
						print index
					print path_to_gts
					print new_gts_name					
					print new_image_name
					print new_src
					#print index_2
					#print new_src
					#print new_image_name
	'''
	if 'gts' in file_list[i]:
		dst = '/media/zhipeng/Windows7_OS/LIDC-annotation/Label_1'
		data_list = os.listdir(file_list[i])
		for j in range(0, len(data_list)):
			path_to_image= file_list[i] + '/' + data_list[j]
			image_list = os.listdir(path_to_image)
			for k in range(0, len(image_list)):
				if '.txt' not in image_list[k]:
					path_to_gts= file_list[i] + '/' + data_list[j] + '/' + image_list[k]
					gts_list = os.listdir(path_to_gts)
					for l in range(0, len(gts_list)):
						if 'id3' in gts_list[l]:
							src = '{}/{}'.format(path_to_gts, gts_list[l])
							#shutil.copy(src, dst)
							new_src = dst + '/' + gts_list[l]
							new_gts_name = '{}/GT_{}.tif'.format(dst, index)
							#os.rename(new_src, new_gts_name)
							index += 1
							print index
							#print src
							#print new_src
							#print new_gts_name
	'''