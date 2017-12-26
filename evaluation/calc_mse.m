function mse = calc_mse( Y, predictions )
% CALC_MSE calculates the mean squared error.
% 
% Requires:
%   Y           - true output values
%   predictions - predicted output values
% 
% Returns:
%   mse         - mean squared error

N = size(Y,1);
error = Y - predictions;
mse = error' * error / N;

end