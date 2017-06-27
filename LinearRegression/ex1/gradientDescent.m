function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
alpha = 0.002;
for iter = 1:10
  % shuffling X
  randidx = randperm(size(X,1));
  X_shuffled = X(randidx(1:size(X)),:);
    for i = 1:m,
    x = X(i,:);
    h = theta(1) + (theta(2)*x(2));

    theta_zero = theta(1) - alpha  * sum(h-y(i));
    theta_one  = theta(2) - alpha  * sum((h - y(i)) .* x(2));

    theta = [theta_zero; theta_one];
    endfor

end

end
