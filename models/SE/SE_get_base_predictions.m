function base_models_preds = SE_get_base_predictions( SE_models, SE_params, X, S )
% SE_GET_BASE_PREDICTIONS infers the predictions for nodes' outputs, made
% by all base components of SE.
% 
% Requires:
%   SE_models         - unstructured predictors for the GCRF components
%                       within SE
%   SE_params         - GCRF components' parameters
%   X                 - nodes' input vectors
%   S                 - similarity matrix describing the correlations among
%                       nodes
% 
% Returns:
%   base_models_preds - N-by-M matrix containing predictions for the
%                       outputs of all N nodes, made by each of the M base
%                       components within SE

N = size(X,1);
M = size(SE_models,1);

% Initialize a matrix for storing the base components' predictions
base_models_preds = zeros(N,M);
for m = 1 : M
    % Get the predictions of the m-th base component
    R = UR_predict(SE_models{m},X);
    base_models_preds(:,m) = GCRF_predict(SE_params(m,:),1,S,R);
end

end