function MSE = calc_MSE( Y, mu )

N = size(Y,1);
error = Y - mu;
MSE = error' * error / N;

end

