import os
import sys
import argparse
import numpy as np


def main(file_path: str):
    ann_path_list = np.loadtxt(file_path, dtype=np.str)
    ann_list = np.vstack([np.loadtxt(path, ndmin=2) for path in ann_path_list])

    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    np.savetxt('tmp/all.txt', ann_list, fmt='%f', delimiter=',')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('list_file', type=str, help='trian annotation file lists , must be [class,x,y,w,h]')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.list_file)
