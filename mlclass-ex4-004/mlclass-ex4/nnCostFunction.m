function [J grad] = nnCostFunction(nn_params, ...
input_layer_size, ...
hidden_layer_size, ...
num_labels, ...
X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X = [ones(m, 1) X];

z2 = Theta1 * X';
A2 = sigmoid(z2);

% Add ones to the A2 data matrix
n = size(A2', 1);
A2 = [ones(n, 1) A2'];
z3 = Theta2 * A2';

hypotheses = sigmoid(z3);
A3 = hypotheses;

Y = zeros(size(A3));

for inum = 1:m
  Y(:, inum) = eye(num_labels)(:,y(inum,:))';
end

J = sum(sum((-Y .* log(hypotheses)) - ((1 - Y) .* log(1 - hypotheses))));

% non-vectorized version
% for inum = 1:m
%   y_vect = eye(num_labels)(:,y(inum,:))';
%   J += (-y_vect * log(hypotheses(:,inum))) - ((1 - y_vect) * log(1 - hypotheses(:,inum)));
% end

t1_reg = sum(sum(Theta1 .^ 2));
t2_reg = sum(sum(Theta2 .^ 2));

t1_reg = t1_reg - sum(Theta1(:,1) .^ 2);
t2_reg = t2_reg - sum(Theta2(:,1) .^ 2);

reg_term = (lambda/(2 * m)) * ((t1_reg) + (t2_reg));

J = 1/m * sum(J) + reg_term;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delta_3 = zeros(size(A3'));

for inum = 1:m
  y_vect = eye(num_labels)(:,y(inum,:))';
  delta_3(inum,:) = A3'(inum, :) .- y_vect;
end

% delta_2 = Theta2'(2:end,:) * delta_3';
% delta_2 = delta_2 .* sigmoidGradient(z2);
delta_2 = Theta2'(2:end,:) * delta_3';
delta_2 = delta_2 .* sigmoidGradient(z2);

Theta1_grad = 1/m * delta_2 * X;
Theta2_grad = 1/m * delta_3' * A2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
