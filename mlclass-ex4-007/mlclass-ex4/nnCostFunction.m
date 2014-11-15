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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%X(4,:)
%Theta1(3,:)
a1 = [ones(m,1) X];

z2 = Theta1*a1';%Theta1*a1';
a2 = sigmoid(z2);
%a2(4:4)
%k = size(a2,1);
a2 = [ones(1,m); a2];
%a2(:,5);
z3 = Theta2*a2;  %10*5000
a3 = sigmoid(z3);
%a3(5,5);
%size(a3)

K = size(a3,1);

th3 = zeros(size(a3));
%size(th3)
for i = 1:K
    J =J - 1/m * sum((y==i).*log(a3(i,:)')+(1-(y==i)).*log(1-a3(i,:)'));
    th3(i,:) = a3(i,:) - (y==i)'; 
end
% regularization term
reg_term = Theta1(:,2:end).*Theta1(:,2:end);
reg_term = sum(sum(reg_term));
reg_term2 = Theta2(:,2:end).*Theta2(:,2:end);
reg_term += sum(sum(reg_term2));
J += reg_term/(2*m);

th2 = (Theta2'*th3)(2:end,:).*sigmoidGradient(z2);

delta2 = th3*a2';


delta1 = th2*a1;

Theta2_grad(:,2:end) = delta2(:,2:end)./m + lambda*Theta2(:,2:end);
Theta2_grad(:,1) = delta2(:,1)./m;

%Theta1_grad
Theta1_grad(:,2:end) = delta1(:,2:end)./m + lambda*Theta1(:,2:end);
Theta1_grad(:,1) = delta1(:,1)./m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
