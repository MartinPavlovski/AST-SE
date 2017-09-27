function [SE_models, SE_params] = SE_train( X, Y, S, M, eta )

N = size(X,1);
train_indices = (1:N)';
Nsub = round(eta*N);

SE_models = cell(M,1);
SE_params = zeros(M,2);

for m = 1 : M
    subset_indices = datasample(train_indices,Nsub,'Replace',false);
    X_subset = X(subset_indices,:);
    Y_subset = Y(subset_indices,:);  
    S_subset = S(subset_indices,subset_indices);
    
    base_model = UR_train(X_subset,Y_subset);
    R_subset = UR_predict(base_model,X_subset);
    theta = GCRF_train(Y_subset,S_subset,R_subset);
    
    SE_models{m} = base_model;
    SE_params(m,:) = theta;
end

end

