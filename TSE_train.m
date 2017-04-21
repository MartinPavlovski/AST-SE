function [TSE_models, TSE_params] = TSE_train( Xtrain, Ytrain, Strain, M, eta )

Ntrain = size(Xtrain,1);

TSE_models = cell(M,1);
TSE_params = zeros(M,2);
train_indices = [1 : Ntrain]';
Nsub = round(eta*Ntrain);


for m = 1 : M
    
    subset_indices = datasample(train_indices,Nsub,'Replace',false);
    X_subset = Xtrain(subset_indices,:);
    Y_subset = Ytrain(subset_indices,:);  
    S_subset = Strain(subset_indices,subset_indices);
    
    base_model = base_train(X_subset,Y_subset);
    R_subset = base_predict(base_model,X_subset);
    theta = GCRF_train(Y_subset,S_subset,R_subset);
    
    TSE_models{m} = base_model;
    TSE_params(m,:) = theta;
    
end

end

