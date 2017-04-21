function [l, delta_theta, mu] = GCRF_objective(Theta,L,R,Y)
%

%likelihood gradient
[n,a] = size(R);
alpha = Theta(1:a); beta = Theta(a+1);

gamma = sum(alpha);
M = beta*L + gamma*eye(n);          %precision matrix
M_inv = inv(M);                     %covariance matrix
b = R*alpha';
mu = M\b;                           %prediction vector
e = Y - mu;                         %error vector

%First Order Alpha Updates
tr = trace(M_inv); 
delta_alpha = zeros(1,a);
for i = 1:a
    delta_alpha(i) = -e'*e + 2*(R(:,i)-mu)'*e + .5*tr;
end

%First Order Beta Updates
tr = trace(M_inv*L);  %#ok<MINV>
delta_beta = -( (Y + mu)' * L * e ) + .5*tr;

%Negative Log-Likelihood
l = -e'*M*e - (1/2)*log(det(M_inv));
l = -l;

delta_theta = [-delta_alpha,-delta_beta];
