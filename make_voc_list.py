import os
import re
import numpy as np
import sys
import argparse


# image_txt_path = '/media/zqh/Datas/DataSet/train.txt'


def main(train_file: str):
    image_path_list = np.loadtxt(train_file, dtype=str)

    if not os.path.exists('data'):
        os.makedirs('data')

    np.savetxt('data/voc_img.list', image_path_list, fmt='%s')

    ann_list = list(image_path_list)
    ann_list = [re.sub(r'JPEGImages', 'labels', s) for s in ann_list]
    ann_list = [re.sub(r'.jpg', '.txt', s) for s in ann_list]
    np.savetxt('data/voc_ann.list', ann_list, fmt='%s')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='trian.txt file path')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args.train_file)
