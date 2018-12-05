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

    % calculate costFunction
      h = sigmoid(X * theta);
      temp = theta(2:end);
      cost = ((-y)' * log(h) - (1 - y)' * log(1 - h));
      temp1 = lambda ./ (2 * m) * (temp' * temp);
      J = (1 ./ m * cost ) + temp1 ;
    
    % calculate grad
  
      temp2 = theta;
      temp2 (1) = 0;
      grad = ((X' * ( h - y)) ./ m) + (lambda /m ) * temp2 ;
    



% =============================================================

end