function loss = WM_objective(w, Y, predictions, lambda)

[N,~] = size(predictions);

loss = (1/N) .* sum( (Y - (predictions * w')).^2 ) + lambda .* sum(abs(w));
