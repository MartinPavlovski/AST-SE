function [neg_ll, delta_theta, mu] = GCRF_objective( theta, L, R, Y )
% GCRF_OBJECTIVE calculates the value of GCRF's objective function.
% 
% Requires:
%   theta       - GCRF parameters
%   L           - Laplacian of a between-node similarity matrix
%   R           - unstructured predictions
%   Y           - vector containing the nodes' output values
% 
% Returns:
%   neg_ll      - negative conditional log-likelihood
%   delta_theta - first order updates (can optionally be used in case
%                 the optimization is converted to an unconstrained
%                 optimization)
%   mu          - vector containing predictions for the outputs of all
%                 nodes

[N,K] = size(R);
alpha = theta(1:K); beta = theta(K+1);

gamma = sum(alpha);
Q = beta*L + gamma*eye(N);          % Precision matrix
Q_inv = inv(Q);                     % Inverse precision matrix
b = R*alpha';
mu = Q\b;                           % Prediction vector
e = Y - mu;                         % Error vector

% Calculate the first order alpha updates
tr = trace(Q_inv);
delta_alpha = zeros(1,K);
for i = 1:K
    delta_alpha(i) = -e'*e + 2*(R(:,i)-mu)'*e + .5*tr;
end

% Calculate the first order beta updates
tr = trace(Q_inv*L);  %#ok<MINV>
delta_beta = -( (Y + mu)' * L * e ) + .5*tr;

% Calculate the negative log-likelihood
neg_ll = -e'*Q*e - (1/2)*log(det(Q_inv));
neg_ll = -neg_ll;

delta_theta = [-delta_alpha, -delta_beta];

end