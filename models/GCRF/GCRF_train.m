function theta = GCRF_train( Y, S, R )
% GCRF_TRAIN trains a Gaussian Conditional Random Field (GCRF) model.
% 
% Requires:
%   Y     - N-by-1 vector containing the nodes' output values
%   S     - N-by-N symmetric similarity matrix describing the correlations
%           among nodes
%   R     - N-by-K matrix containing predictions for the outputs of all N
%           nodes, made by K unstructured predictors
% 
% Returns:
%   theta - optimal GCRF parameters

% Get the size of R (N outputs, K predictors)
[N,K] = size(R);
% Normalize S
S = (S/sum(sum(S))) * N;
% Calculate the Laplacian of S
L = diag(sum(S)) - S;

% Initialize GCRF's parameters
alpha = ones(1,K); beta = 0; theta = [alpha,beta];
% Set optimization constraints
A = [ones(1,K) , 0; ...
     zeros(1,K) , 1];
b = [0; 0];
A = -A;

% Optimize GCRF's objective
options = optimset('Algorithm','interior-point','MaxIter',1000,'GradObj','off','TolX',1e-6,'TolCon',1e-6,'Display','Off');
% Find the optimal GCRF parameters
theta = fmincon(@(theta)GCRF_objective(theta,L,R,Y),theta,A,b,[],[],[],[],[],options);

end