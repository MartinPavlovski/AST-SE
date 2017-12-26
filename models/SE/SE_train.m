function [SE_models, SE_params] = SE_train( X, Y, S, M, eta )
% SE_TRAIN trains a Structured Ensemble (SE).
% 
% Requires:
%   X         - nodes' input vectors
%   Y         - nodes' output values
%   S         - similarity matrix describing the correlations among nodes
%   M         - number of GCRF components within SE
%   eta       - subsampling fraction
% 
% Returns:
%   SE_models - unstructured predictors for the GCRF components within SE
%   SE_params - GCRF components' parameters

N = size(X,1);
train_indices = (1:N)';
Nsub = round(eta*N);

% Create arrays for storing SE's unstructured predictors and their GCRF
% parameters
SE_models = cell(M,1);
SE_params = zeros(M,2);

for m = 1 : M
    % Sample a subnetwork from the original network
    subset_indices = datasample(train_indices,Nsub,'Replace',false);
    X_subset = X(subset_indices,:);
    Y_subset = Y(subset_indices,:);  
    S_subset = S(subset_indices,subset_indices);
    
    % Train a GCRF on the sampled subnetwork
    base_model = UR_train(X_subset,Y_subset);
    R_subset = UR_predict(base_model,X_subset);
    theta = GCRF_train(Y_subset,S_subset,R_subset);
    
    % Store the current GCRF component, i.e. the current unstructured
    % predictor along with its corresponding GCRF parameters
    SE_models{m} = base_model;
    SE_params(m,:) = theta;
end

end