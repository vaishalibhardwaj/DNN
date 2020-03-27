import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a

def relu(z):
    a=z*(z>0)
    return a

def softmax(z):
    e=np.exp(z)
    a=e/np.sum(e,axis=0,keepdims=True)
    assert(a.shape==e.shape)
    return a

def init_params(layer_dims, init_type='random', scale=0.01,):
    parameters={}
    L=len(layer_dims)
    for l in range(1,L):
        if init_type=='random':
            parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) *scale
        elif init_type=='zeros':
            parameters['W'+str(l)] = np.zeros((layer_dims[l],layer_dims[l-1]))
        elif init_type=='he':
            parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * ((2/layer_dims[l-1])**0.5)
        else:
            raise ValueError()

        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
    return parameters

    def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b

    if activation=='sigmoid':
        A = sigmoid(Z)
    elif activation=='relu':
        A = relu(Z)
    elif activation=='tanh':
        A = np.tanh(Z)
    elif activation=='softmax':
        A = softmax(Z)
    else:
        raise ValueError('The type of nn layer should be sigmoid, relu, tanh or softmax !!!')

    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = {'W':W, 'b':b, 'Z':Z, 'A':A, 'A_prev':A_prev}
    return cache


def forword_propagation(X, parameters, layer_dims, layer_types):
    L = len(layer_types)
    caches=[]
    A_prev = X
    for l in range(L):
        W = parameters['W'+str(l+1)]
        b = parameters['b'+str(l+1)]
        cache = linear_activation_forward(A_prev, W, b, layer_types[l])
        A_prev = cache['A']
        caches.append(cache)
    Yhat = caches[L-1]['A']
    assert(Yhat.shape==(layer_dims[-1], X.shape[1]))
    return Yhat, caches

def linear_activation_backward(dA, cache, activation, lambd):
    A_prev = cache['A_prev']
    A = cache['A']
    W = cache['W']
    m = A.shape[1]
    if activation=='sigmoid':
        dZ = A*(1-A)*dA
    elif activation=='relu':
        dZ = (A>0)*dA
    elif activation=='tanh':
        dZ = (1-A**2)*dA
    elif activation=='softmax':
        """
        pd--partial derivative
        pdA[i,j] / pdZ[i,j] = A[i,j] * (1 - A[i,j]
        pdA[k,j] / pdZ[i,j] = - A[i,j] * A[k,j],    k!=i
        dZ[i,j] = - A[i,j] * Sigma(dA[k,j] * A[k,j]) + A[i,j] * dA[i,j]
        """
        dZ = - A * np.sum(dA*A, axis=0, keepdims=True) + A*dA
    else:
        raise ValueError('The type of nn layer should be sigmoid, relu, tanh or softmax !!!')

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m

    if lambd != 0.0:
        dW = dW + lambd/m * W
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backword_propagation(X, Y, parameters, caches, layer_dims, layer_types, lambd):
    L = len(layer_types)
    grads={}
    A = caches[L-1]['A']
    if layer_types[-1]=='sigmoid':
        dA = - np.divide(Y, A) + np.divide(1 - Y, 1 - A)
    elif layer_types[-1]=='softmax':
        dA = - np.divide(Y,A)
    else:
        raise ValueError('The type of output layer should be sigmoid or softmax !!!')

    grads['dA'+str(L)] = dA

    for l in reversed(range(L)):
        cache=caches[l]
        dA_prev, dW, db=linear_activation_backward(grads['dA'+str(l+1)], cache, layer_types[l], lambd)
        grads['dA'+str(l)] = dA_prev
        grads['dW'+str(l+1)] = dW
        grads['db'+str(l+1)] = db
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters


def compute_cost(Yhat, Y, parameters, activation, lambd):
    m = Y.shape[1]
    cost = 0
    if activation=='sigmoid':
        Yhat[np.where(Yhat==0)] = 1e-8    # in case that log(0)
        Yhat[np.where(Yhat==1)] = 1-1e-8    # in case that log(0)
        cost = -np.sum(Y*np.log(Yhat)+(1-Y)*np.log(1-Yhat))/m
    elif activation=='softmax':
        Yhat[np.where(Yhat==0)] = 1e-8    # in case that log(0)
        cost = -np.sum(Y*np.log(Yhat))/m

    else:
        raise ValueError('The type of output layer should be sigmoid or softmax !!!')

    cost = np.squeeze(cost)    #make sure cost.ndim=1
    if lambd != 0:
        L = len(parameters) // 2
        regul = 0
        for l in range(L):
            regul = regul + np.sum(np.square(parameters["W" + str(l+1)]))    #L2 regularization
        cost = cost + lambd * regul *0.5/m
    return cost


def model(X, Y, hidden_layer_dims, layer_types, learning_rate, num_iterations, num_batches=1, lambd=0.0 , init_type='random',decay_rate=1):
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])
    parameters = init_params(layer_dims, init_type)
    costs=[]
    batch_size = Y.shape[1] // num_batches
    for i in range(num_iterations):
        for j in range(num_batches):
            # Forward propagation
            Yhat,caches = forword_propagation(X[:, j*batch_size : (j+1)*batch_size], parameters, layer_dims, layer_types)

            # Compute cost
            cost = compute_cost(Yhat, Y[:, j*batch_size : (j+1)*batch_size], parameters, layer_types[-1], lambd)

            # Backward propagation
            grads = backword_propagation(X[:, j*batch_size : (j+1)*batch_size], Y[:, j*batch_size : (j+1)*batch_size], parameters, caches, layer_dims, layer_types, lambd)

            # Update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

            if i % 10 == 0:
                print ("Cost after iteration %d and batch %d: %f" %(i, j, cost))
                costs.append(cost)

        learning_rate = learning_rate * decay_rate

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()

    return parameters

def predict(X, Y, parameters, hidden_layer_dims, layer_types, threshold=0.5):
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])
    Yhat, _ = forword_propagation(X, parameters, layer_dims, layer_types)
    if layer_types[-1]=='sigmoid':
        Y_predict = (Yhat>threshold)*1
        accuracy = np.sum(Y_predict ==Y) / Y.shape[1]

    elif layer_types[-1]=='softmax':
        Y_predict = np.argmax(Yhat,axis=0).reshape((1,-1))
        accuracy = np.sum(np.squeeze(Y_predict) == np.argmax(Y, axis=0)) / Y.shape[1]
    else:
        raise ValueError('The type of output layer should be sigmoid or softmax !!!')


    return Y_predict, accuracy

def forword_propagation_with_dropout(X, parameters, layer_dims, layer_types, keep_prob = 0.5):
    L = len(layer_types)
    caches=[]
    A_prev = X
    for l in range(L):
        W = parameters['W'+str(l+1)]
        b = parameters['b'+str(l+1)]
        cache = linear_activation_forward(A_prev, W, b, layer_types[l])
        if l != (L-1):
            D = np.random.rand(cache['A'].shape[0], cache['A'].shape[1])
            D = (D > keep_prob) * 1
            cache['A'] =  cache['A'] * D / keep_prob
            cache['D'] = D
        A_prev = cache['A']
        caches.append(cache)
    Yhat = caches[L-1]['A']
    assert(Yhat.shape==(layer_dims[-1], X.shape[1]))
    return Yhat, caches

def backword_propagation_with_dropout(X, Y, parameters, caches, layer_dims, layer_types, keep_prob = 0.5):
    L = len(layer_types)
    grads={}
    A = caches[L-1]['A']
    if layer_types[-1]=='sigmoid':
        dA = - np.divide(Y, A) + np.divide(1 - Y, 1 - A)
    elif layer_types[-1]=='softmax':
        dA = - np.divide(Y,A)
    else:
        raise ValueError('The type of output layer should be sigmoid or softmax !!!')

    grads['dA'+str(L)] = dA

    for l in reversed(range(L)):
        cache=caches[l]
        dA_prev, dW, db=linear_activation_backward(grads['dA'+str(l+1)], cache, layer_types[l], lambd =0.0)
        if l != 0:
            dA_prev = dA_prev * caches[l-1]['D'] / keep_prob
        grads['dA'+str(l)] = dA_prev
        grads['dW'+str(l+1)] = dW
        grads['db'+str(l+1)] = db
    return grads

def model_with_dropout(X, Y, hidden_layer_dims, layer_types, learning_rate, num_iterations, num_batches = 1, keep_prob = 0.5, init_type='random' ):
    """
    dropout should not be applied to input layer (layer 0) and outputlayer (the last layer)
    """
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])
    parameters = init_params(layer_dims)
    costs=[]
    batch_size = Y.shape[1] // num_batches
    for i in range(num_iterations):
        for j in range(num_batches):
            # Forward propagation
            Yhat,caches = forword_propagation_with_dropout(X[:, j*batch_size : (j+1)*batch_size], parameters, layer_dims, layer_types)

            # Compute cost
            cost =compute_cost(Yhat, Y[:, j*batch_size : (j+1)*batch_size], parameters, layer_types[-1], lambd=0)

            # Backward propagation
            grads = backword_propagation_with_dropout(X[:, j*batch_size : (j+1)*batch_size], Y[:, j*batch_size : (j+1)*batch_size], parameters, caches, layer_dims, layer_types)

            # Update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example
            if i % 10 == 0:
                print ("Cost after iteration %d and batch %d: %f" %(i, j, cost))
                costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v['dW'+str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        v['db'+str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        s['dW'+str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        s['db'+str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, learning_rate, beta1, beta2, t):
    epsilon = 1e-8
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(L):
        v['dW'+str(l+1)] = beta1 * v['dW'+str(l+1)] + (1-beta1) * grads["dW" + str(l+1)]
        v['db'+str(l+1)] = beta1 * v['db'+str(l+1)] + (1-beta1) * grads["db" + str(l+1)]
        v_corrected['dW'+str(l+1)] = v['dW'+str(l+1)] / (1 - beta1**t)
        v_corrected['db'+str(l+1)] = v['db'+str(l+1)] / (1 - beta1**t)
        s['dW'+str(l+1)] = beta2 * s['dW'+str(l+1)] + (1-beta2) * (grads["dW" + str(l+1)]**2)
        s['db'+str(l+1)] = beta2 * s['db'+str(l+1)] + (1-beta2) * (grads["db" + str(l+1)]**2)
        s_corrected['dW'+str(l+1)] = s['dW'+str(l+1)] / (1 - beta2**t)
        s_corrected['db'+str(l+1)] = s['db'+str(l+1)] / (1 - beta2**t)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected['dW'+str(l+1)] / (s_corrected['dW'+str(l+1)]**0.5 + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected['db'+str(l+1)] / (s_corrected['db'+str(l+1)]**0.5 + epsilon)
    return parameters, v, s

def model_with_adam(X, Y, hidden_layer_dims, layer_types, learning_rate, num_iterations, num_batches=1, lambd=0.0 , init_type='random',beta1=0.9, beta2=0.999, t=2, decay_rate=0.99 ):
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])
    parameters = init_params(layer_dims, init_type)
    v, s = initialize_adam(parameters)
    costs=[]
    batch_size = Y.shape[1] // num_batches
    for i in range(num_iterations):
        for j in range(num_batches):
            # Forward propagation
            Yhat,caches = forword_propagation(X[:, j*batch_size : (j+1)*batch_size], parameters, layer_dims, layer_types)

            # Compute cost
            cost = compute_cost(Yhat, Y[:, j*batch_size : (j+1)*batch_size], parameters, layer_types[-1], lambd)

            # Backward propagation
            grads = backword_propagation(X[:, j*batch_size : (j+1)*batch_size], Y[:, j*batch_size : (j+1)*batch_size], parameters, caches, layer_dims, layer_types, lambd)

            # Update parameters
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, learning_rate, beta1, beta2, t)

            if i % 10 == 0:
                print ("Cost after iteration %d and batch %d: %f" %(i, j, cost))
                costs.append(cost)

        learning_rate = learning_rate * decay_rate

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def model_with_dropout_adam(X, Y, hidden_layer_dims, layer_types, learning_rate, num_iterations, num_batches = 1, keep_prob = 0.5, init_type='random',beta1=0.9, beta2=0.999, t=2, decay_rate=0.99 ):
    """
    dropout should not be applied to input layer (layer 0) and outputlayer (the last layer)
    :param X: input data of shape (size of input layer, number of examples)
    :param Y: labels vector  of shape (size of output layer, number of examples),
              ---for multi claasification, Y must use ont-hot encoding
    :param hidden_layer_dims: python list containing the size of each hidden layer
    :param layer_types: python list containing the type of each layer: "sigmoid" or "relu" "tanh"
    :param learning_rate: scalar
    :param num_iterations: scalar
    :param num_batches: scalar
    :param keep_prob: scalar, dropout parameter
    :return: parameters: python dictionary containing weight matrix and bias vector, {'Wi': , 'bi': }
    """
    layer_dims=[X.shape[0],]
    if hidden_layer_dims!=None:
        layer_dims.extend(hidden_layer_dims)
    layer_dims.append(Y.shape[0])
    parameters = init_params(layer_dims)
    v, s = initialize_adam(parameters)
    costs=[]
    batch_size = Y.shape[1] // num_batches
    for i in range(num_iterations):
        for j in range(num_batches):
            # Forward propagation
            Yhat,caches = forword_propagation_with_dropout(X[:, j*batch_size : (j+1)*batch_size], parameters, layer_dims, layer_types)

            # Compute cost
            cost =compute_cost(Yhat, Y[:, j*batch_size : (j+1)*batch_size], parameters, layer_types[-1], lambd=0)

            # Backward propagation
            grads = backword_propagation_with_dropout(X[:, j*batch_size : (j+1)*batch_size], Y[:, j*batch_size : (j+1)*batch_size], parameters, caches, layer_dims, layer_types)

            # Update parameters
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, learning_rate, beta1, beta2, t)

            if i % 10 == 0:
                print ("Cost after iteration %d and batch %d: %f" %(i, j, cost))
                costs.append(cost)

        learning_rate = learning_rate * decay_rate

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters