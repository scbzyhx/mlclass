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

for row = 1:m

    J = J - y(row)*log(sigmoid(X(row,:)*theta));
    J = J - (1-y(row))*log(1-sigmoid(X(row,:)*theta));
    grad +=  (sigmoid(X(row,:)*theta) - y(row))*X(row,:)';
    
    
endfor
J = J/m + sum(theta(2:size(theta)).*theta(2:size(theta)))*lambda/(2*m);
%J = J/m + sum(theta.*theta)*lambda/(2*m);
grad = grad/m;

grad(2:size(theta)) += lambda*theta(2:size(theta))/m;



% =============================================================

end
