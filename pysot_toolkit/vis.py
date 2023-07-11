# import cv2
# import numpy as np
# import os
#
# save_dir=""
# img_path=""
# gt_path=""
# videos = os.listdir(img_path)
# gt = os.listdir(gt_path)
# assert len(videos) == len(gt)
# # imgs = sorted([imgs[i] for i in range(len(imgs)) if imgs[i].split('.')[1]=='jpg'],key=lambda x:x.split('.')[0])
# for i,video in enumerate(videos):
#     if video != "car1_1":
#         continue
#     gt_txt_path=os.path.join(gt_path,video)
#     gt_txt=np.loadtxt(gt_txt_path+'.txt', delimiter=',')
#     imgs = os.listdir(os.path.join(img_path,video))
#     if not os.path.exists(os.path.join(save_dir,video)):
#         os.makedirs(os.path.join(save_dir,video))
#     imgs = sorted([imgs[i] for i in range(len(imgs)) if imgs[i].split('.')[1] == 'jpg'], key=lambda x: x.split('.')[0])
#     for idx, img in enumerate(imgs):
#         print(img)
#         gt_bbox = gt_txt[idx]
#         if str(gt_bbox[0])=="nan" or str(gt_bbox[1])=='nan' or str(gt_bbox[2])=='nan' or str(gt_bbox[3])=='nan':
#             continue
#         img_read = cv2.imread(os.path.join(img_path,video,img))
#         cv2.rectangle(img_read, (int(gt_bbox[0]), int(gt_bbox[1])),
#                     (int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])),
#                     (0, 255, 0), 3)
#         cv2.imwrite(os.path.join(save_dir,img),img_read)


import imageio
import os
data_list = sorted(os.listdir(os.path.join('/home/zxh/pic/bike1')))

def compose_gif():
    gif_images = []
    for i in range(200):
        gif_images.append(imageio.imread(os.path.join('/home/zxh/pic/bike1',data_list[i])))
    imageio.mimsave('/home/zxh/project/ESTrack.gif', gif_images, duration=50)
compose_gif()

# import cv2
# #(1080,1920)
# img=cv2.imread('/home/zxh/pic/pic_0.jpg')
# cv2.line(img, (30,40),(1090,2000),(0,255,255),4)
# cv2.imshow('image',img)
# cv2.waitKey(0)