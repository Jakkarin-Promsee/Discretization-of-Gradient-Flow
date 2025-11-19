import numpy as np

def flatten_params(layers):
    flat_list = []
    shapes = []

    for layer in layers:
        if hasattr(layer, "W"):
            shapes.append(("W", layer.W.shape))
            flat_list.append(layer.W.ravel())
        if hasattr(layer, "b"):
            shapes.append(("b", layer.b.shape))
            flat_list.append(layer.b.ravel())

    return np.concatenate(flat_list), shapes

def unflatten_params(theta_vec, layers, shapes):
    idx = 0
    for layer in layers:
        if hasattr(layer, "W"):
            name, shape = shapes[idx]
            size = np.prod(shape)
            layer.W = theta_vec[idx:idx+size].reshape(shape)
            idx += size
        if hasattr(layer, "b"):
            name, shape = shapes[idx]
            size = np.prod(shape)
            layer.b = theta_vec[idx:idx+size].reshape(shape)
            idx += size