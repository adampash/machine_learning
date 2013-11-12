function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% calculate hypotheses
hypothesis = sigmoid(X * theta);

% calculate non-regularized cost
reg = (lambda / (2 * m)) * sum(theta .^ 2);
reg1 = (lambda / (2 * m)) * (theta(1) .^ 2);
reg2 = reg - reg1;

J = (-1/m) * sum((y .* log(hypothesis)) + ((1 - y) .* log(1 - hypothesis))) + reg2;

% this is the gradient for use w/the fminunc function
reg = (lambda / m) .* theta;
reg(1) = 0;

% apply regularization param to all but theta[0]
grad = (1/m) * (X' * (hypothesis - y)) + reg;

% =============================================================

end
