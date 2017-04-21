function mu = GCRF_predict( theta, K, S, R )

N = size(S,1);
S = (S/sum(sum(S))) * N;
L = diag(sum(S)) - S;
alpha = theta(1:K); gamma = sum(alpha); beta = theta(end);
Q = beta*L + gamma*eye(N);
mu = Q\(R * alpha');


end

