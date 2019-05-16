import numpy as np
from tools.utils import Helper
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import sys
import argparse
import tensorflow as tf


def tf_fake_iou(X: tf.Tensor, centroids: tf.Tensor) -> tf.Tensor:
    """ calc the fake iou between x and centroids

    Parameters
    ----------
    X : tf.Tensor
        dataset array, shape = [?,5,2]
    centroids : tf.Tensor
        centroids,shape = [?,5,2]

    Returns
    -------
    tf.Tensor
        iou score, shape = [?,5]
    """
    s1 = X[..., 0] * X[..., 1]
    s2 = centroids[..., 0] * centroids[..., 1]
    iner = tf.minimum(X[..., 0], centroids[..., 0]) * tf.minimum(X[..., 1], centroids[..., 1])
    iou_score = 1 - iner / (s1 + s2 - iner)
    return iou_score


def findClosestCentroids(X: tf.Tensor, centroids: tf.Tensor) -> tf.Tensor:
    """ find close centroids

    Parameters
    ----------
    X : tf.Tensor
        dataset array, shape = [?,5,2]
    centroids : tf.Tensor
        centroids array, shape = [?,5,2]

    Returns
    -------
    tf.Tensor
        idx, shape = [?,]    
    """
    idx = tf.argmin(tf_fake_iou(X, centroids), axis=1)
    return idx


def computeCentroids(X: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """ use idx calc the new centroids

    Parameters
    ----------
    X : np.ndarray
        shape = [?,2]
    idx : np.ndarray
        shape = [?,]
    k : int
        the centroids num

    Returns
    -------
    np.ndarray
        new centroids
    """
    m, n = np.shape(X)
    centroids = np.zeros((k, n))
    for i in range(k):
        centroids[i, :] = np.mean(X[np.nonzero(idx == i)[0], :], axis=0)
    return centroids


def plotDataPoints(X, idx, K):
    plt.scatter(X[:, 0], X[:, 1], c=idx)


def plotProgresskMeans(X, centroids_history, idx, K, i):
    plotDataPoints(X, idx, K)
    # Plot the centroids as black x's
    for i in range(len(centroids_history) - 1):
        plt.plot(centroids_history[i][:, 0], centroids_history[i][:, 1], 'rx')
        plt.plot(centroids_history[i + 1][:, 0], centroids_history[i + 1][:, 1], 'bx')
        # Plot the history of the centroids with lines
        for j in range(K):
            # matplotlib can't draw line like [x1,y1] to [x2,y2]
            # it have to write like [x1,x2] to [y1,y2] f**k!
            plt.plot(np.r_[centroids_history[i + 1][j, 0], centroids_history[i][j, 0]],
                     np.r_[centroids_history[i + 1][j, 1], centroids_history[i][j, 1]], 'k--')
    # Title
    plt.title('Iteration number {}'.format(i + 1))


def tile_x(x: np.ndarray, k: int):
    # tile the array
    x = x[:, np.newaxis, :]
    x = np.tile(x, (1, k, 1))
    return x


def tile_c(initial_centroids: np.ndarray, m: int):
    c = initial_centroids[np.newaxis, :, :]
    c = np.tile(c, (m, 1, 1))
    return c


def build_kmeans_graph(new_x: np.ndarray, new_c: np.ndarray):
    """ build calc kmeans graph

    Parameters
    ----------
    new_x : np.ndarray
        shape= [?,5,2]
    new_c : np.ndarray
        shape = [?,5,2]

    Returns
    -------
    tuple
    in_x : x placeholder
    in_c : c placeholder
    out_idx : output idx tensor, shape [?,]
    """
    in_x = tf.placeholder(tf.float64, shape=np.shape(new_x), name='in_x')
    in_c = tf.placeholder(tf.float64, shape=np.shape(new_c), name='in_c')
    out_idx = findClosestCentroids(in_x, in_c)

    return in_x, in_c, out_idx


def runkMeans(X: np.ndarray, initial_centroids: np.ndarray, max_iters: int,
              plot_progress=False):
    # init value
    m, _ = X.shape
    k, _ = initial_centroids.shape

    # history list
    centroid_history = []

    # save history
    centroids = initial_centroids.copy()
    centroid_history.append(centroids.copy())

    # build tensorflow graph
    new_x, new_c = tile_x(X, k), tile_c(initial_centroids, m)
    assert new_x.shape == new_c.shape
    in_x, in_c, idx = build_kmeans_graph(new_x, new_c)

    """ run kmeans """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    for i in range(max_iters):
        idx_ = sess.run(idx, feed_dict={in_x: new_x, in_c: new_c})
        new_centrois = computeCentroids(X, idx_, k)
        centroid_history.append(new_centrois.copy())
        new_c = tile_c(new_centrois, m)

    sess.close()
    if plot_progress:
        plt.figure()
        plotProgresskMeans(X, centroid_history, idx_, k, max_iters)
        plt.show()

    return new_centrois, idx_


def main(train_set: str, outfile: str, max_iters: int, in_hw: tuple, out_hw: tuple, is_random: bool, is_plot: bool):
    X = np.loadtxt(train_set, delimiter=',')
    x = X[:, 3:]  # x= [w,h]
    if is_random == 'True':
        initial_centroids = np.random.rand(5, 2)
    else:
        initial_centroids = np.vstack((np.linspace(0.05, 0.3, num=5), np.linspace(0.05, 0.5, num=5)))
        initial_centroids = initial_centroids.T
    centroids, idx = runkMeans(x, initial_centroids, 10, is_plot)
    centroids /= np.array([1 / out_hw[1], 1 / out_hw[0]])
    centroids = np.array(sorted(centroids, key=lambda x: (x[0])))
    if np.any(np.isnan(centroids)):
        print('\033[1;31m' + 'ERROR' + '\033[0m' + ' run out the ' + '\033[1;33m' + 'NaN' +
              '\033[0m' + ' value please ' + '\033[1;32m' + 'ReRun!' + '\033[0m')
    else:
        np.savetxt(outfile, centroids, fmt='%f')
        print('\033[1;35m' + 'SUCCESS' + '\033[0m' + ' save file to ' + '\033[1;33m' + outfile +
              '\033[0m')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('train_set', type=str, help='trian file lists , must be [class,x,y,w,h]')
    parser.add_argument('--max_iters', type=int, help='kmeans max iters', default=10)
    parser.add_argument('--is_random', type=str, help='wether random generate the center', choices=['True', 'False'], default='False')
    parser.add_argument('--is_plot', type=str, help='wether show the figure', choices=['True', 'False'], default='True')
    parser.add_argument('--in_hw', type=int, help='net work input image size', default=(240, 320), nargs='+')
    parser.add_argument('--out_hw', type=int, help='net work output image size', default=(7, 10), nargs='+')
    parser.add_argument('out_anchor_file', type=str, help='output anchors list file, name must be xxx_anchors.list')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.train_set, args.out_anchor_file, args.max_iters, args.in_hw, args.out_hw, args.is_random, args.is_plot)
