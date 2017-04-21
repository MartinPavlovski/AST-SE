function predictions = TRE_get_predictions( TSE_models, TSE_params, X, S )

N = size(X,1);
M = size(TSE_models,1);

predictions = zeros(N,M);
for m = 1 : M
    R = base_predict(TSE_models{m},X);
    predictions(:,m) = GCRF_predict(TSE_params(m,:), 1, S, R);
end

end

