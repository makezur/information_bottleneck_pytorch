import numpy as np


def get_aligned_representations(representations, order):
    for epoch in range(len(representations)):
        for layer in range(len(representations[0])):
            representations[epoch][layer] = representations[epoch][layer][np.argsort(order[epoch]), :]

    return representations
