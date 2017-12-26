% MAIN contains code for experimental evaluation of the proposed model and
% its alternatives, on the H3N2 Virus Influenza network. All models are
% briefly described in the following:
%   - AST-SE (proposed): Adaptive Skip-Train Structured Ensemble, a
%     sampling-based structured regression ensemble for prediction on top
%     of temporal networks.
%   - LR: An L1-regularized linear regression. LR was employed as an
%     unstructured predictor for each of the following models in order to
%     achieve efficiency.
%   - GCRF: Standard GCRF model that enables the chosen unstructured
%     predictor to learn the network structure.
%   - SE: Structured ensemble composed of multiple GCRF models.
%   - WSE: Weighted structured ensemble that combines the predictions of
%     multiple GCRFs in a weighted mixture in order to predict the nodes'
%     outputs in the next timestep.

clc; clearvars; close all;

% Add the paths to the sub-directories containing data, models'
% implementations and evaluation measures
addpath('data')
addpath(genpath('models'))
addpath('evaluation')

% Load the H3N2 Virus Influenza dataset
load('H3N2_data.mat')
T = size(XList,1);

% Initialize models' parameters
M = 30;
eta = 0.3;
lambda = 0.1;

models_names = {'LR','GCRF','SE','WSE','AST-SE'};
num_models = size(models_names,2);
test_mses = -1*ones(T-1,num_models);

% Create arrays for storing AST-SE's components (i.e. the unstructured
% predictors accompanied by their corresponding GCRF parameters), their
% weights, and the data, from previous timesteps
ASTSE_models_prev = -1;
ASTSE_params_prev = -1;
ASTSE_w_prev = -1;
X_prev = -1;
Y_prev = -1;
S_prev = -1;

% Seed the random number generator
rng(1, 'twister');

for t = 1 : T - 1
    
    % Construct the training data by taking the inputs, outputs and
    % similarity matrix from timestep t
    X_train = XList{t};
    Y_train = YList{t};
    S_train = S;
    
    % Use the data from t+1 for testing (one-step-ahead prediction)
    X_test = XList{t+1};
    Y_test = YList{t+1};
    S_test = S;
    
    
    % Train and test an unstructured regressor (UR) - in this case LR is 
    % used as a UR
    base_model = UR_train(X_train, Y_train);
    Rtrain = UR_predict(base_model,X_train);
    Rtest = UR_predict(base_model,X_test);
    UR_test_mse = calc_mse(Y_test, Rtest);
    
    
    % Train and test a GCRF
    theta = GCRF_train(Y_train,S_train,Rtrain);
    GCRF_preds = GCRF_predict(theta, 1, S_test, Rtest);
    GCRF_test_mse = calc_mse(Y_test, GCRF_preds);
    
    
    % Train and test an SE
    [SE_models, SE_params] = SE_train(X_train, Y_train, S_train, M, eta);
    predictions_test = SE_get_base_predictions(SE_models, SE_params, X_test, S_test);
    SE_preds = mean(predictions_test,2);
    SE_test_mse = calc_mse(Y_test, SE_preds);
    
    
    % Train and test a WSE
    predictions_train = SE_get_base_predictions(SE_models, SE_params, X_train, S_train);
    w = get_weights(Y_train, predictions_train, lambda);
    WSE_preds = predictions_test * w';
    WSE_test_mse = calc_mse(Y_test, WSE_preds);
    
    
    % Train and test the proposed AST-SE
    if t > 1
        X_prev = XList{t-1};
        Y_prev = YList{t-1};
        S_prev = S;
    end    
    [ASTSE_models, ASTSE_params, ASTSE_w, state, chooses_min] = ASTSE_train(X_prev, Y_prev, S_prev, XList{t}, YList{t}, S, ...
                                                                            ASTSE_models_prev, ASTSE_params_prev, ASTSE_w_prev, M, eta, lambda);
    ASTSE_models_prev = ASTSE_models;
    ASTSE_params_prev = ASTSE_params;
    ASTSE_w_prev = ASTSE_w;
    
    ASTSE_preds = SE_get_base_predictions(ASTSE_models, ASTSE_params, X_test, S_test);
    ASTSE_preds = ASTSE_preds * ASTSE_w';
    ASTSE_test_mse = calc_mse(Y_test, ASTSE_preds);
	
    
    % Store the testing MSEs
    test_mses(t,:) = [UR_test_mse, GCRF_test_mse, SE_test_mse, WSE_test_mse, ASTSE_test_mse];
    fprintf('(t=%d)\t%f\t%f\t%f\t%f\t%f\n', t, UR_test_mse, GCRF_test_mse, SE_test_mse, WSE_test_mse, ASTSE_test_mse);
    
end


fprintf('\n\n');


% Display the average MSEs of all models along with the corresponding
% confidence intervals
avg_test_mses = -1*ones(num_models,1);
conf_int_widths = -1*ones(num_models,1);
for col = 1 : num_models
    [mse_avg, mse_int_width] = calc_conf_interval(test_mses(:,col));
    avg_test_mses(col) = mse_avg;
    conf_int_widths(col) = mse_int_width;
end

fprintf('Average Testing MSEs:\n');
res_table = table(avg_test_mses,conf_int_widths,'RowNames',models_names);
res_table.Properties.VariableNames = {'Average_MSE' 'Conf_interval_width'};
res_table

