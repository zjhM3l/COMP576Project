__author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork  # Import NeuralNetwork

class Layer:
    """
    This class represents a single layer in the network, providing methods for feedforward and backpropagation.
    """
    def __init__(self, input_dim, output_dim, actFun_type='tanh', seed=0):
        np.random.seed(seed)
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        self.actFun_type = actFun_type

    def actFun(self, z, type):
        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif type == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError("Unsupported activation function type")

    def diff_actFun(self, z, type):
        if type == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif type == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif type == 'relu':
            return (z > 0).astype(float)
        else:
            raise ValueError("Unsupported activation function type")

    def feedforward(self, X):
        self.z = X.dot(self.W) + self.b
        self.a = self.actFun(self.z, self.actFun_type)
        return self.a

    def backprop(self, delta, X, next_W=None):
        if next_W is not None:
            delta = delta.dot(next_W.T) * self.diff_actFun(self.z, self.actFun_type)
        dW = X.T.dot(delta)
        db = np.sum(delta, axis=0, keepdims=True)
        return delta, dW, db

class DeepNeuralNetwork(NeuralNetwork):
    """
    This class builds and trains a deep neural network with n layers.
    """
    def __init__(self, layer_dims, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param layer_dims: list containing dimensions of each layer
        :param actFun_type: activation function for hidden layers. Options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        super().__init__(layer_dims[0], layer_dims[1], layer_dims[-1], actFun_type, reg_lambda, seed)
        self.layers = []
        self.reg_lambda = reg_lambda
        np.random.seed(seed)
        # Create each layer and add to the list
        for i in range(1, len(layer_dims)):
            self.layers.append(Layer(layer_dims[i-1], layer_dims[i], actFun_type, seed))

    def feedforward(self, X):
        '''
        Perform feedforward across all layers.
        :param X: input data
        :return: probabilities
        '''
        self.a_list = [X]
        a = X
        for layer in self.layers:
            a = layer.feedforward(a)
            self.a_list.append(a)
        exp_scores = np.exp(a)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def calculate_loss(self, X, y):
        '''
        Calculate the loss for the prediction.
        :param X: input data
        :param y: given labels
        :return: the loss
        '''
        num_examples = len(X)
        self.feedforward(X)
        corect_logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        # Add regularization
        for layer in self.layers:
            data_loss += self.reg_lambda / 2 * np.sum(np.square(layer.W))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        Predict the labels for the given data.
        :param X: input data
        :return: predicted labels
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        Perform backpropagation through all layers.
        :param X: input data
        :param y: given labels
        :return: gradients for each layer
        '''
        num_examples = len(X)
        delta = self.probs
        delta[range(num_examples), y] -= 1
        delta /= num_examples
        grads = []
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            input_a = self.a_list[i]
            if i == len(self.layers) - 1:
                delta, dW, db = layer.backprop(delta, input_a)
            else:
                next_W = self.layers[i + 1].W
                delta, dW, db = layer.backprop(delta, input_a, next_W)
            grads.insert(0, (dW, db))
        return grads

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        Train the model using backpropagation and gradient descent.
        :param X: input data
        :param y: given labels
        :param epsilon: learning rate
        :param num_passes: number of passes through the dataset
        :param print_loss: whether to print the loss during training
        '''
        for i in range(0, num_passes):
            self.feedforward(X)
            grads = self.backprop(X, y)
            # Update weights and biases
            for j in range(len(self.layers)):
                dW, db = grads[j]
                self.layers[j].W += -epsilon * (dW + self.reg_lambda * self.layers[j].W)
                self.layers[j].b += -epsilon * db
            # Optionally print the loss
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y, title, filename):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :param title: title for the plot
        :param filename: filename to save the plot
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y, title, filename)

def generate_data(dataset='moons'):
    '''
    generate data
    :param dataset: type of dataset ('moons' or 'iris')
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    if dataset == 'moons':
        X, y = datasets.make_moons(200, noise=0.20)
    elif dataset == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    else:
        raise ValueError("Unsupported dataset type")
    return X, y

def plot_decision_boundary(pred_func, X, y, title, filename):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :param title: title for the plot
    :param filename: filename to save the plot
    :return:
    '''
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(filename)
    plt.show()

def main():
    # Generate and visualize Make-Moons dataset
    X, y = generate_data('moons')
    
    # Train and visualize with different number of layers
    configurations = [
        ([2, 3, 2], 'tanh'),
        ([2, 5, 5, 2], 'relu'),
        ([2, 10, 10, 5, 2], 'sigmoid')
    ]
    
    for layer_dims, actFun_type in configurations:
        print(f"Training with configuration: Layers={layer_dims}, Activation Function={actFun_type}")
        model = DeepNeuralNetwork(layer_dims, actFun_type=actFun_type)
        model.fit_model(X, y)
        model.visualize_decision_boundary(X, y, title=f"Decision Boundary with Layers {layer_dims} and {actFun_type} Activation Function (Moons)", filename=f"deep_decision_boundary_moons_{actFun_type}_layers_{len(layer_dims)}.png")

    # Generate and visualize Iris dataset
    X, y = generate_data('iris')
    X = X[:, :2]  # Use only the first two features for visualization
    
    # Train and visualize with different number of layers on Iris dataset
    configurations = [
        ([2, 5, 3], 'tanh'),
        ([2, 10, 3], 'relu'),
        ([2, 7, 5, 3], 'sigmoid')
    ]
    
    for layer_dims, actFun_type in configurations:
        print(f"Training with configuration: Layers={layer_dims}, Activation Function={actFun_type} (Iris)")
        model = DeepNeuralNetwork(layer_dims, actFun_type=actFun_type)
        model.fit_model(X, y)
        model.visualize_decision_boundary(X, y, title=f"Decision Boundary with Layers {layer_dims} and {actFun_type} Activation Function (Iris)", filename=f"deep_decision_boundary_iris_{actFun_type}_layers_{len(layer_dims)}.png")

if __name__ == "__main__":
    main()