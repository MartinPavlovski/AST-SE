function theta = GCRF_train(Y,S,R)
%

[n,a] = size(R);         %n outputs, a predictors
tic
S = (S/sum(sum(S))) * n; %normalize S
L = diag(sum(S)) - S;    %derive Laplacian of S

%initialize
alpha = ones(1,a); beta = 0; theta = [alpha,beta];
%constraing functions 
A = [ones(1,a) , 0; zeros(1,a) , 1]; b = [0; 0]; A = -A;

tic
%Run Gradient Descent
options = optimset('Algorithm','interior-point','MaxIter',1000,'GradObj','off','TolX',1e-6,'TolCon',1e-6,'Display','Off');
%options = optimset(options,'UseParallel','always');
theta = fmincon(@(theta)GCRF_objective(theta,L,R,Y),theta,A,b,[],[],[],[],[],options);
