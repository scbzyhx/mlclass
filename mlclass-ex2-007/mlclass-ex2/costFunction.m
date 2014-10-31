function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta


for row = 1:m

    J = J - y(row)*log(sigmoid(X(row,:)*theta));
    J = J - (1-y(row))*log(1-sigmoid(X(row,:)*theta));
    grad +=  (sigmoid(X(row,:)*theta) - y(row))*X(row,:)';
    
    
endfor
J = J/m;
grad = grad/m;








% =============================================================

end

% J = J - (y(row)*log(sigmoid(X(row,:)*theta)) + (1-y(row))*log(sigmoid(1-X(row,:)*theta)));
    
%    J = J - y(row)*log(X(row,:)*theat);
%    J = J - 
%    if y(row) == 1
%	cost = 0 - log(X(row,:)*theta);
%    else
%        cost = 0 - log(1-(X(row,:)*theta));
%    endif
%    J = J + cost;
%