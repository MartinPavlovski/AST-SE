function [neg_ll, delta_theta, mu] = GCRF_objective( theta, L, R, Y )

[N,K] = size(R);
alpha = theta(1:K); beta = theta(K+1);

gamma = sum(alpha);
M = beta*L + gamma*eye(N);          % precision matrix
M_inv = inv(M);                     % inverse covariance matrix
b = R*alpha';
mu = M\b;                           % prediction vector
e = Y - mu;                         % error vector

% first order alpha updates
tr = trace(M_inv); 
delta_alpha = zeros(1,K);
for i = 1:K
    delta_alpha(i) = -e'*e + 2*(R(:,i)-mu)'*e + .5*tr;
end

% first order beta updates
tr = trace(M_inv*L);  %#ok<MINV>
delta_beta = -( (Y + mu)' * L * e ) + .5*tr;

% negative log-likelihood
neg_ll = -e'*M*e - (1/2)*log(det(M_inv));
neg_ll = -neg_ll;

delta_theta = [-delta_alpha, -delta_beta];

end