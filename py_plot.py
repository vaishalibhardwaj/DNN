import matplotlib.pyplot as plt
import numpy as np
import math

def show_decision_boundry(X, Y, Y_onehot, predict_function, parameters, hidden_layer_dims, layer_types, h = 0.02, space = 1):
    x1_min = X[0, :].min() - space
    x1_max = X[0, :].max() + space
    x2_min = X[1, :].min() - space
    x2_max = X[1, :].max() + space
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))
    X_meshgird = np.c_[x1.ravel(), x2.ravel()]  # flatten and oncatenate along the second axis
    X_meshgird = X_meshgird.T
    Y_meshgird, _ = predict_function(X_meshgird, Y_onehot, parameters, hidden_layer_dims, layer_types)
    Y_meshgird = Y_meshgird.reshape(x1.shape)
    fig = plt.figure()
    plt.contourf(x1, x2, Y_meshgird, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.show()


def display_image_samples(datasets, image_shape, sample_indices):
    num_samples = len(sample_indices)
    num_axis1 = math.floor(math.sqrt(num_samples))
    num_axis2 = num_samples // num_axis1
    for i in range(num_axis1):
        for j in range(num_axis2):
            index = sample_indices[num_axis2*i+j]
            image=datasets[:, index].reshape(image_shape)
            plt.subplot(num_axis1, num_axis2 , num_axis2*i+j+1)
            plt.imshow(image)
    plt.show()