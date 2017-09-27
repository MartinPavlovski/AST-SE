function w = get_weights( Y, predictions, lambda )

[~,M] = size(predictions);

% initialize weights uniformly
w = (1/M)*ones(1,M);

% set constraints
A = [eye(M) ; -eye(M)]; b = [ones(M,1) ; zeros(M,1)];
Aeq = ones(1,M); beq = 1;

% determine weights
options = optimset('Algorithm','interior-point','MaxIter',1000,'GradObj','off','TolX',1e-6,'TolCon',1e-6,'Display','Off');
w = fmincon(@(w)reg_quad_loss(w,Y,predictions,lambda),w,A,b,Aeq,beq,[],[],[],options);

end

