function [all_theta] = oneVsAll(X, y, num_labels,lambda, anpha, num_iters)  

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
[all_theta] = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
  
 
  for i = 1:num_labels
    Y = (y==i);
    initial_theta = zeros(n+1, 1);
    [theta] = gradientDescentMulti(X, Y,initial_theta, anpha,lambda, num_iters);
    all_theta(i, :) = theta;
  endfor
      









% =========================================================================


endfunction
