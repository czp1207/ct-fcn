import os
import shutil

#os.mkdir('/media/zhipeng/Windows7_OS/LIDC-annotation/label_1')
#os.mkdir('/media/zhipeng/Windows7_OS/LIDC-annotation/label_2')
#os.mkdir('/media/zhipeng/Windows7_OS/LIDC-annotation/label_3')
#os.mkdir('/media/zhipeng/Windows7_OS/LIDC-annotation/label_4')

gts_name = os.listdir('/media/zhipeng/Windows7_OS/LIDC-annotation/gts')

#for i in range(0, len(gts_name)):
	#os.mkdir('/media/zhipeng/Windows7_OS/LIDC-annotation/label_1/{}'.format(gts_name[i]))
	#os.mkdir('/media/zhipeng/Windows7_OS/LIDC-annotation/label_2/{}'.format(gts_name[i]))
	#os.mkdir('/media/zhipeng/Windows7_OS/LIDC-annotation/label_3/{}'.format(gts_name[i]))
	#os.mkdir('/media/zhipeng/Windows7_OS/LIDC-annotation/label_4/{}'.format(gts_name[i]))
for i in range(0, len(gts_name)):
	path_to_gts = '/media/zhipeng/Windows7_OS/LIDC-annotation/gts' + '/' + gts_name[i]
	gts_ls = os.listdir(path_to_gts)
	print gts_ls[0]
	for j in range(0, len(gts_ls)):
		if '.txt' not in gts_ls[j]:
			gts = os.listdir(path_to_gts + '/' + gts_ls[j])
			if (len(gts) != 0):
				for k in range(0, len(gts)):
					if 'id_3' in gts[k]:
						shutil.move('{}/{}/{}'.format(path_to_gts, gts_ls[j], gts[k]), '/media/zhipeng/Windows7_OS/LIDC-annotation/label_3/{}'.format(gts_name[i]))