function base_models_preds = SE_get_base_predictions( SE_models, SE_params, X, S )

N = size(X,1);
M = size(SE_models,1);

base_models_preds = zeros(N,M);
for m = 1 : M
    R = UR_predict(SE_models{m},X);
    base_models_preds(:,m) = GCRF_predict(SE_params(m,:),1,S,R);
end

end

