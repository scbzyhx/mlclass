function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%h = 1./(1 + exp(0 - X*theta));
h = X*theta;
J = sum((h - y).*(h-y))/(2*m) + sum(theta(2:end).*theta(2:end))*lambda/(2*m);

dim = size(X,2);
for i = 2:dim
   grad(i) = sum((h - y).*(X(:,i)))/m + theta(i)*lambda/m;
end

%grad(2) = sum((h - y).*(X(:,2)))/m + theta(2)*lambda/m;
grad(1) = sum((h - y).*(X(:,1)))/m;





% =========================================================================

grad = grad(:);

end
