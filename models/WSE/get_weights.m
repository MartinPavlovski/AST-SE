function w = get_weights( Y, predictions, lambda )
% GET_WEIGHTS finds the weights of an ensemble's components.
% 
% Requires:
%   Y           - true output values
%   predictions - predicted output values
%   lambda      - nonnegative regularization parameter
% 
% Returns:
%   w           - components' weights

[~,M] = size(predictions);

% Initialize weights uniformly
w = (1/M)*ones(1,M);

% Set optimization constraints
A = [eye(M) ; -eye(M)]; b = [ones(M,1) ; zeros(M,1)];
Aeq = ones(1,M); beq = 1;

% Find weights
options = optimset('Algorithm','interior-point','MaxIter',1000,'GradObj','off','TolX',1e-6,'TolCon',1e-6,'Display','Off');
w = fmincon(@(w)reg_quad_loss(w,Y,predictions,lambda),w,A,b,Aeq,beq,[],[],[],options);

end