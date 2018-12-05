                           
    % Load Training Data from Folder Images
    
    X = [];
    y = [];
    index = 1;
    for k = 0 : 9
        dir = pwd(); %quay tro lai thu muc làm viec hien tai
        dir = strcat(dir, '\So0\*.png');
        files = glob(dir);
        for i = 1:numel(files)
          f = files{i};
          J = imagePreProcess(f);
          X = [X; J(:)'];
          y = [y;10];
          %I = imread(f);
          %X = [X; I(:)'];
        endfor
    endfor

    data = [X, y];
    m = size(X, 1);
    
    % Randomly select 100 data points to display
    rand_indices = randperm(m);
    sel = X(rand_indices(1:25), :);
    displayData(sel);
    
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    
    % Compute and display initial cost and gradient for regularized logistic
% regression
    initial_theta = zeros(size(X, 2), 1);
    lambda = 1;
    [cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
    fprintf('Cost at initial theta (zeros): %f\n', cost);
    fprintf(' %f \n', grad);
##    
##    %fprintf('Running gradient descent ...\n');
##     X= [ones(m, 1) X]; 
##    % Initialize fitting parameters
##    initial_theta = zeros(size(X, 2), 1);
##    % Choose some alpha value
##    lambda = 1;
##    anpha = 1;
##    num_iters = 300;
##    % Init Theta and Run Gradient Descent 
##    [theta,cost]= gradientDescentMulti(X, y, initial_theta, anpha, lambda, num_iters);
##    fprintf('Cost at theta found by gradient Descent: %f\n', cost);
##    fprintf('theta: \n');
##    fprintf(' %f \n', theta);
    
   %% One-vs-All Training 
    fprintf('\nTraining One-vs-All Logistic Regression...\n') 
    lambda = 0.1;
    anpha = 3;
    num_iters= 2300;
    num_labels = 10; 

    [all_theta] = oneVsAll(X, y, num_labels, lambda, anpha, num_iters);


    %% Predict for One-Vs-All 
    pred = predictOneVsAll(all_theta, X);

    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);