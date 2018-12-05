function [theta, cost] = gradientDescentMulty(X, y, theta, anpha,lambda, num_iters)
 
 % Initialize some useful values
  m = length(y);  % number of training examples
  
  for iter = 1:num_iters
     temp = theta;
     temp(1) = 0;
    
    h = sigmoid(X * theta);
    grad = 1/m * X' * ( h - y);
    theta = theta - anpha*grad;    
    theta = theta - anpha *1/m * lambda * temp;
  endfor
  [cost, grad] = costFunctionReg(theta, X, y, lambda);
endfunction

    