import os
import numpy as np
import skimage
import shutil

if __name__ == "__main__":
    fddb_path = '/home/zhengqihang/FDDB'

    fddb_fold_path = os.path.join(fddb_path, 'FDDB-folds')
    fddb_label_path = os.path.join(fddb_path, 'labels')
    if os.path.exists(fddb_label_path):
        shutil.rmtree(fddb_label_path)
    os.mkdir(fddb_label_path)

    anns_path_list = [os.path.join(fddb_fold_path, name) for name in os.listdir(fddb_fold_path) if 'ellipseList' in name]

    anns_list = []
    for name in anns_path_list:
        with open(name) as f:
            l = f.read().splitlines()
        anns_list.extend(l)

    img_path_list = []
    label_path_list = []
    img_flag = True
    cnt = 0
    for i, line in enumerate(anns_list):
        if img_flag:
            if 'img' in line:
                img_path = fddb_path+'/'+line+'.jpg'
                img_path_list.append(img_path)
                img_flag = False
        else:
            anns = []
            for j in range(int(line)):
                anns.append(anns_list[i+1+j].split())
            anns_array = np.asfarray(anns)
            anns_array[:, -1] -= 1
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
            anns_array = anns_array[:, [5, 3, 4, 1, 0]]
            # anns_array[:, 3:5] *= 2
            img_src = skimage.io.imread(img_path)
            anns_array[:, [1, 3]] /= img_src.shape[1]
            anns_array[:, [2, 4]] /= img_src.shape[0]

            img_flag = True
            label_path = os.path.join(fddb_label_path, '{:06d}.txt'.format(cnt))
            np.savetxt(label_path, anns_array, fmt='%f')
            cnt += 1
            label_path_list.append(label_path)
            # with open(os.path.join(fddb_label_path, '{:06d}.txt'.format(cnt)), 'w') as f:
    if not os.path.exists('data'):
        os.makedirs('data')
    np.savetxt('data/fddb_ann.list', label_path_list, fmt='%s')
    np.savetxt('data/fddb_img.list', img_path_list, fmt='%s')
