function R = UR_predict( base_model, X )
% UR_PREDICT predicts output values, given a matrix of input vectors.
% 
% Requires:
%   base_model - coefficients of an L1-regularized linear regression
%   X          - input vectors
% 
% Returns:
%   R          - predicted outputs

R = X*base_model;

end