function loss = reg_quad_loss( w, Y, predictions, lambda )

[N,~] = size(predictions);

% Regularized quadratic loss
loss = (1/N) .* sum( (Y - (predictions * w')).^2 ) + lambda .* sum(abs(w));

end
