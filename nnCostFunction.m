function [J, grad] = nnCostFunction(nn_params, ...
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

% Add ones to the X data matrix
X = [ones(m, 1) X];

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

% size of X (a1)     = 5000 x 401
% size of Theta1     = 25 x 401
% size of Theta2     = 10 x 26
% size of z2         = 5000 x 25
% size of a2         = 5000 x 25
% size of a2 (add 1) = 5000 x 26
% size of z3         = 5000 x 10
% size of a3         = 5000 x 10

% creating vector y with binary numbers 1 and 0
I = eye(num_labels);
Y = zeros(m,num_labels);
for i=1:m
  Y(i,:)= I(y(i),:);
end

disp('***** size of new Y *****')
disp(size(Y));
% size of Y = 5000 x 10

% feedforward calculation
a1 = X;
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% assigning h(theta)
hx = a3;
disp(size(hx));

% do not penlize theta(1)
tempTheta1 = Theta1;
tempTheta1(1) = 0;
tempTheta2 = Theta2;
tempTheta2(1) = 0;

% cost calculation
tempJ = Y.*log(hx) + (1 - Y).*log(1 - hx);

disp('**** size of tempJ ****');
disp(size(tempJ));
% size of tempJ = 5000 x 10

% J = (-1/m)*sum(sum(tempJ)) + ...
%     (lambda/(2*m))*(sum(sum(tempTheta1.^2)) + sum(sum(tempTheta2.^2)));

J = (-1/m)*sum(sum(tempJ,2)) + ...
   (lambda/(2*m))*(sum(sum(tempTheta1.^2)) + sum(sum(tempTheta2.^2)));

disp('**** size of J ****')
disp(size(J));

% calculate back-propogation error
partialDelta_3 = a3 - Y;
% size of partialDelta_3 = 5000 x 10
z = (partialDelta_3*Theta2);
partialDelta_2 = z(:,2:end).*sigmoidGradient(z2);
% size of partialDelta_2 = 5000 x 25

% accumulate gradient 
delta_1 = partialDelta_2'*a1;
% size of delta_1 = 25 x 401
delta_2 = partialDelta_3'*a2;
% size of delta_2 = 10 x 26

% divide accumulated gradient by 1/m
D1 = (1/m)*(delta_1 + lambda*tempTheta1);
% size of D1 = 25 x 401
D2 = (1/m)*(delta_2 + lambda*tempTheta2);
% size of D2 = 10 x 26

Theta1_grad = D1;
Theta2_grad = D2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
