function mse = calc_mse( Y, predictions )

N = size(Y,1);
error = Y - predictions;
mse = error' * error / N;

end

