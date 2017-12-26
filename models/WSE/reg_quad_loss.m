function loss = reg_quad_loss( w, Y, predictions, lambda )
% REG_QUAD_LOSS calculates the regularized quadratic loss.
% 
% Requires:
%   w           - weights of an ensemble's components
%   Y           - true output values
%   predictions - predicted output values
%   lambda      - nonnegative regularization parameter
% 
% Returns:
%   loss        - regularized quadratic loss

[N,~] = size(predictions);

loss = (1/N) .* sum( (Y - (predictions * w')).^2 ) + lambda .* sum(abs(w));

end