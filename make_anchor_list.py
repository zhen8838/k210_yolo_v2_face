import numpy as np
from tools.utils import Helper
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import sys
import argparse


def _fake_iou(wh, wh1):
    s1 = wh[0]*wh[1]
    s2 = wh1[0]*wh1[1]
    iner = np.minimum(wh[0], wh1[0])*np.minimum(wh[1], wh1[1])
    return 1 - iner/(s1+s2-iner)


def findClosestCentroids(X: np.ndarray, centroids: np.ndarray):
    # Set K
    K = np.size(centroids, 0)
    # You need to return the following variables correctly.
    idx = np.zeros((np.size(X, 0), 1))

    idx = np.argmin(cdist(X, centroids, metric=_fake_iou), axis=1)

    return idx  # 1d array


def computeCentroids(X: np.ndarray, idx: np.ndarray, K: int):
    m, n = np.shape(X)
    centroids = np.zeros((K, n))
    for i in range(K):
        centroids[i, :] = np.mean(X[np.nonzero(idx == i)[0], :], axis=0)
    return centroids


def plotDataPoints(X, idx, K):
    plt.scatter(X[:, 0], X[:, 1], c=idx)


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    plotDataPoints(X, idx, K)
    # Plot the centroids as black x's
    plt.plot(previous[:, 0], previous[:, 1], 'rx')
    plt.plot(centroids[:, 0], centroids[:, 1], 'bx')
    # Plot the history of the centroids with lines
    for j in range(np.size(centroids, 0)):
        # matplotlib can't draw line like [x1,y1] to [x2,y2]
        # it have to write like [x1,x2] to [y1,y2] f**k!
        plt.plot(np.r_[centroids[j, 0], previous[j, 0]],
                 np.r_[centroids[j, 1], previous[j, 1]], 'k--')

    # Title
    plt.title('Iteration number {}'.format(i))


def runkMeans(X: np.ndarray, initial_centroids: np.ndarray, max_iters: int,
              plot_progress=False):
    # Plot the data if we are plotting progress
    if plot_progress:
        plt.figure()

    # Initialize values
    m, n = np.shape(X)
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids.copy()
    previous_centroids = centroids.copy()
    idx = np.zeros((m, 1))

    # Run K-Means
    for i in range(max_iters):
        # Output progress
        print('K-Means iteration {}/{}...\n'.format(i, max_iters))
        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)
        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids.copy()
        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)
    if plot_progress:
        plt.show()

    return centroids, idx


def main(train_set: str, outfile: str, max_iters: int, in_hw: tuple, out_hw: tuple, is_random: bool, is_plot: bool):
    helper = Helper('data/{}_img.list'.format(train_set), 'data/{}_ann.list'.format(train_set), None, None, in_hw, out_hw)
    g = helper.generator(is_training=False, is_make_lable=False)
    _, true_box = next(g)
    X = true_box.copy()
    try:
        while True:
            _, true_box = next(g)
            X = np.vstack((X, true_box))
    except StopIteration as e:
        print('collotation all box')
    x = X[:, 3:]
    initial_centroids = np.vstack((np.linspace(0.05, 0.3, num=5), np.linspace(0.05, 0.5, num=5)))
    initial_centroids = initial_centroids.T
    # initial_centroids = np.random.rand(5, 2)
    centroids, idx = runkMeans(x, initial_centroids, 10, is_plot)
    centroids /= np.array([helper.grid_w, helper.grid_h])
    np.savetxt(outfile, centroids, fmt='%f')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_set',      type=str,   help='trian file lists',                choices=['voc', 'coco', 'fddb'], default='fddb')
    parser.add_argument('--max_iters',      type=int,   help='kmeans max iters',                default=10)
    parser.add_argument('--is_random',      type=bool,  help='wether random generate the center', default=False)
    parser.add_argument('--is_plot',        type=bool,  help='wether show the figure',          default=True)
    parser.add_argument('--in_hw',          type=int,   help='net work input image size',       default=(240, 320), nargs='+')
    parser.add_argument('--out_hw',         type=int,   help='net work output image size',      default=(7, 10), nargs='+')
    parser.add_argument('--out_anchor_file', type=str,   help='output anchors list file path')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.train_set, args.out_anchor_file, args.max_iters, args.in_hw, args.out_hw, args.is_random, args.is_plot)
