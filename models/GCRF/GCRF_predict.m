function mu = GCRF_predict( theta, K, S, R )
% GCRF_PREDICT predicts the output values of all nodes in a network.
% 
% Requires:
%   theta - GCRF parameters
%   K     - number of unstructured predictors
%   S     - similarity matrix describing the correlations among nodes
%   R     - unstructured predictions
% 
% Returns:
%   mu    - vector containing predictions for the outputs of all nodes

N = size(S,1);
S = (S/sum(sum(S))) * N;
L = diag(sum(S)) - S;
alpha = theta(1:K); gamma = sum(alpha); beta = theta(end);
Q = beta*L + gamma*eye(N);
mu = Q\(R * alpha');

end