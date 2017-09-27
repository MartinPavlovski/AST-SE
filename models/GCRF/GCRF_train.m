function theta = GCRF_train( Y, S, R )

[N,K] = size(R);         % N outputs, K predictors
S = (S/sum(sum(S))) * N; % normalize S
L = diag(sum(S)) - S;    % calculate the Laplacian of S

% initialize parameters
alpha = ones(1,K); beta = 0; theta = [alpha,beta];
% set constraints
A = [ones(1,K) , 0; ...
     zeros(1,K) , 1];
b = [0; 0];
A = -A;

% optimize GCRF's objective
options = optimset('Algorithm','interior-point','MaxIter',1000,'GradObj','off','TolX',1e-6,'TolCon',1e-6,'Display','Off');
theta = fmincon(@(theta)GCRF_objective(theta,L,R,Y),theta,A,b,[],[],[],[],[],options);

end