"""
Calculate the information in the network
Can be by the full distribution rule (for small netowrk) or bt different approximation method
"""
from multiprocessing import cpu_count
from joblib import Parallel, delayed

import warnings
import numpy as np
import numba


NUM_CORES = cpu_count()
warnings.filterwarnings("ignore")


@numba.jit
def entropy(probs):
    return -np.sum(probs * np.ma.log2(probs))


@numba.jit
def joint_entropy(unique_inverse_x, unique_inverse_y, bins_x, bins_y):

    joint_distribution = np.zeros((bins_x, bins_y))
    np.add.at(joint_distribution, (unique_inverse_x, unique_inverse_y), 1)
    joint_distribution /= np.sum(joint_distribution)

    return entropy(joint_distribution)


@numba.jit
def layer_information(layer_output, bins, py, px, unique_inverse_x, unique_inverse_y):
    ws_epoch_layer_bins = bins[np.digitize(layer_output, bins) - 1]
    ws_epoch_layer_bins = ws_epoch_layer_bins.reshape(len(layer_output), -1)

    unique_t, unique_inverse_t, unique_counts_t = np.unique(
        ws_epoch_layer_bins, axis=0,
        return_index=False, return_inverse=True, return_counts=True
    )

    pt = unique_counts_t / np.sum(unique_counts_t)

    # # I(X, Y) = H(Y) - H(Y|X)
    # # H(Y|X) = H(X, Y) - H(X)

    x_entropy = entropy(px)
    y_entropy = entropy(py)
    t_entropy = entropy(pt)

    x_t_joint_entropy = joint_entropy(unique_inverse_x, unique_inverse_t, px.shape[0], layer_output.shape[0])
    y_t_joint_entropy = joint_entropy(unique_inverse_y, unique_inverse_t, py.shape[0], layer_output.shape[0])

    return {
        'local_IXT': t_entropy + x_entropy - x_t_joint_entropy,
        'local_ITY': y_entropy + t_entropy - y_t_joint_entropy
    }


@numba.jit
def calc_information_for_epoch(epoch_number, ws_epoch, bins, unique_inverse_x,
                               unique_inverse_y, pxs, pys):
    """Calculate the information for all the layers for specific epoch"""
    information_epoch = []

    for i in range(len(ws_epoch)):
        information_epoch_layer = layer_information(
            layer_output=ws_epoch[i],
            bins=bins,
            unique_inverse_x=unique_inverse_x,
            unique_inverse_y=unique_inverse_y,
            px=pxs, py=pys
        )
        information_epoch.append(information_epoch_layer)
    information_epoch = np.array(information_epoch)

    print('Processed epoch {}'.format(epoch_number))

    return information_epoch


@numba.jit
def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
    pys = np.sum(label, axis=0) / float(label.shape[0])

    unique_x, unique_x_indices, unique_inverse_x, unique_x_counts = np.unique(
        x, axis=0,
        return_index=True, return_inverse=True, return_counts=True
    )

    pxs = unique_x_counts / np.sum(unique_x_counts)

    # pys_given_x = []
    # for i in range(0, len(unique_x)):
    #     indexs = unique_inverse_x == i
    #     py_x_current = np.mean(label[indexs, :], axis=0)
    #     pys_given_x.append(py_x_current)
    #
    # pys_given_x = np.array(pys_given_x)

    unique_array_y, unique_y_indices, unique_inverse_y, unique_y_counts = np.unique(
        label, axis=0,
        return_index=True, return_inverse=True, return_counts=True
    )
    return pys, None, unique_x, unique_inverse_x, unique_inverse_y, pxs


def get_information(ws, x, label, num_of_bins, every_n=1,
                    return_matrices=False):
    """
    Calculate the information for the network for all the epochs and all the layers

    ws.shape =  [n_epoch, n_layers, n_params]
    ws --- outputs of all layers for all ephocs
    """

    print('Start calculating the information...')

    bins = np.linspace(-1, 1, num_of_bins)
    label = np.array(label).astype(np.float)
    pys, _, unique_x, unique_inverse_x, unique_inverse_y, pxs = extract_probs(label, x)

    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        information_total = parallel(
            delayed(calc_information_for_epoch)(
                i, epoch_output, bins, unique_inverse_x, unique_inverse_y, pxs, pys
            ) for i, epoch_output in enumerate(ws) if i % every_n == 0
        )

    if not return_matrices:
        return information_total

    ixt_matrix = np.zeros((len(information_total), len(ws[0])))
    ity_matrix = np.zeros((len(information_total), len(ws[0])))

    for epoch, layer_info in enumerate(information_total):
        for layer, info in enumerate(layer_info):
            ixt_matrix[epoch][layer] = info['local_IXT']
            ity_matrix[epoch][layer] = info['local_ITY']

    return ixt_matrix, ity_matrix

    # information_total = []
    # for i, epoch_output in enumerate(ws):
    #     information_total.append(
    #         calc_information_for_epoch(i, epoch_output, bins, unique_inverse_x, unique_inverse_y, pxs, pys))
    #
    # return information_total
