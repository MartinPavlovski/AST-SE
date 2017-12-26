function base_model = UR_train( X, Y )
% UR_TRAIN trains an unstructured predictor, more precisely, an
% L1-regularized linear regression model.
% 
% Requires:
%   X          - input vectors
%   Y          - output values
% 
% Returns:
%   base_model - fitted coefficients of an L1-regularized linear regression

base_model = lasso(X,Y,'Lambda',0.1);

end