from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # 1st layer of the neural network: 
        #    - Produces weighted input data with added bias using the linear function: H = (X . W1) + b1
        #    - Passes the outputs of F1 through the ReLU activation function: H = ReLU(H) = max{0, H} 
        #    - Activation functions transform layer output to a non-linear function, which is required to
        #      allow learning meaningful loss gradients (that depend on inputs) when backpropagating errors.
        #      ReLU is used because it's computationally efficient and very effective in practice.
        hidden_layer = np.maximum(0,np.dot(X, W1) + b1)
        
        # 2nd layer of the neural network:
        #    - Weights the output of hidden_layer(h) with bias offset by computing: S = (h . W2) + b2
        #    - The 2nd layer adds more complexity to the model by applying weights and biases to the non-linear
        #      output of hidden_layer. This improves & refines the adjusting of input weights during training,
        #      as backpropagation provides more details on how each input feature affects the loss gradient.
        scores = np.dot(hidden_layer, W2) + b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Compute scores as probabilities using the softmax activation function, where each value (N,C) in the
        # prob_scores (P) matrix represents the probability that class C is the correct label/classification for
        # input data sample N. (Softmax is a popular effective output for non-binary classification problems)
        exp_scores = np.exp(scores)
        prob_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Compute the negative log likelihood of the correct class label for each input (cross-entropy loss).
        # This^ is defined for each input i as follows: L[i] = -log(P[i,y[i]]) ---> range of i = [0,N)
        #    - P[i,y[i]] indicates the probability of input i having the truthful class label (y[i])
        #    - Why we use -log(P): more efficient to differentiate and minimize by SGD, compared to raw probability
        correct_class_loss = -np.log(prob_scores[range(N),y])
        
        # scalar quantity describing average loss over all input data samples
        average_loss = np.sum(correct_class_loss) / N
        
        # Regularization function computed using L2 norm of weight matrices:
        #   - Solves the problem of overfitting model on training data
        #   - Does this^ by penalizing complexity of the model
        #   - L2 norm achieves this by encouraging weight spread out across all neurons/values in a layer
        #   - Lower L2 norm = better weight distribution in W# matrix = less complex layer = lower relative loss from W#
        loss_regularization = reg*np.sum(W1*W1) + reg*np.sum(W2*W2)
        
        # Compute total loss for current model output: quantifies our unhapiness with scores across training data
        loss = average_loss + loss_regularization
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # This section uses chain rule to compute gradients of loss function w.r.t. weight matrices and bias vectors,
        # to quantify how much each layer is negatively affecting the total loss.
        
        # Compute dL/dS[i], where L is the total loss and S[i] represents the class scores for input i.
        # Formula has been derived in Tutorial 2:
        #    dL/dS[i] = dl/dL[i] . dL[i]/dP . dP/dS[i] = (P[i] - 1) / N
        scores_grad = prob_scores
        scores_grad[range(N),y] -= 1
        scores_grad /= N
        
        # In backpropagation:
        #   - Addition operator acts as a "gradient distributor" (all inputs & outputs have identical gradients)
        #   - Multiply operator acts as a "gradient switcher" (value of other operand multiplies the output gradient)
        
        # S = (h . W2) + b2 --> Therefore:
        #   - By gradient distributor, dL/db2 = dL/dS = sum_over_i(dL/dS[i]) = sum_over_rows(scores_grad)
        grads['b2'] = np.sum(scores_grad, axis=0, keepdims=True)
        #   - By gradient switcher, dL/dW2 = (h.Transpose) . (dL/dS)
        #   - Need to transpose h: (cols in M1 = rows in M2) must be true for (M1 . M2) to be defined
        #   - Shape(dL/dW2) = (H,C) = (H,N).(N,C)
        grads['W2'] = np.dot(hidden_layer.T, scores_grad)
        
        # By gradient switcher and matrix product definition, dL/dh = (dL/dS) . (W2.Transpose)
        # Shape(dL/dh) = (N,H) = (N,C).(C,H)
        hidden_grad = np.dot(scores_grad, W2.T)
        # ReLU activation ignores gradients less than zero
        hidden_grad[hidden_layer <= 0] = 0
        
        # By gradient distributor, dL/db1 = dL/dh = sum_over_rows(hidden_grad)
        grads['b1'] = np.sum(hidden_grad, axis=0, keepdims=True)
        # By gradient switcher and matrix product definition, dL/dW1 = (X.Transpose) . (dL/dh)
        # Shape(dL/dW1) = (D,H) = (D,N).(N,H)
        grads['W1'] = np.dot(X.T, hidden_grad)
       
        # Regularization Loss (rL) contributes to the Wn loss gradients in the following way:
        #   - Matrix Addition: Wn = Wn + rL = Wn + reg*(Wn^2)
        #   - Total Wn Gradient: dL/dWn = dL/dWn + d/dWn(reg*(Wn^2)) = (dL/dWn) + (2*reg*Wn)
        grads['W2'] += 2*reg*W2
        grads['W1'] += 2*reg*W1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
