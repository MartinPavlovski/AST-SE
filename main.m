clc; clearvars; close all;

addpath('data')
addpath(genpath('models'))
addpath('evaluation')
 
load('H3N2_data.mat')
T = size(XList,1);

M = 30;
eta = 0.3;
lambda = 0.1;

models_names = {'LR','GCRF','SE','WSE','AST-SE'};
num_models = size(models_names,2);
test_mses = -1*ones(T-1,num_models);

ASTSE_models_prev = -1;
ASTSE_params_prev = -1;
ASTSE_w_prev = -1;
X_prev = -1;
Y_prev = -1;
S_prev = -1;

rng(1, 'twister');
    
for t = 1 : T - 1
    
    X_train = XList{t};
    Y_train = YList{t};
    S_train = S;
    
    X_test = XList{t+1};
    Y_test = YList{t+1};
    S_test = S;
    
    
	% LR: A regularized Linear Regression model is used as an UR (unstructured regressor)
    base_model = UR_train(X_train, Y_train);
    Rtrain = UR_predict(base_model,X_train);
    Rtest = UR_predict(base_model,X_test);
    UR_test_mse = calc_mse(Y_test, Rtest);
    
    
    % GCRF: Standard Gaussian Conditional Random Field
    theta = GCRF_train(Y_train,S_train,Rtrain);
    GCRF_preds = GCRF_predict(theta, 1, S_test, Rtest);
    GCRF_test_mse = calc_mse(Y_test, GCRF_preds);
    
    
    % SE: Structured Ensemble composed of multiple GCRF models
    [SE_models, SE_params] = SE_train(X_train, Y_train, S_train, M, eta);
    predictions_test = SE_get_base_predictions(SE_models, SE_params, X_test, S_test);
    SE_preds = mean(predictions_test,2);
    SE_test_mse = calc_mse(Y_test, SE_preds);
    
    
    % WSE: Weighted Structured Ensemble
    predictions_train = SE_get_base_predictions(SE_models, SE_params, X_train, S_train);
    w = get_weights(Y_train, predictions_train, lambda);
    WSE_preds = predictions_test * w';
    WSE_test_mse = calc_mse(Y_test, WSE_preds);
    
    
    % AST-SE: Adaptive Skip-Train Structured Ensemble
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
	
    
    % Store testing MSEs
    test_mses(t,:) = [UR_test_mse, GCRF_test_mse, SE_test_mse, WSE_test_mse, ASTSE_test_mse];
    fprintf('(t=%d)\t%f\t%f\t%f\t%f\t%f\n', t, UR_test_mse, GCRF_test_mse, SE_test_mse, WSE_test_mse, ASTSE_test_mse);
    
end


fprintf('\n\n');


% Display average MSEs along with the corresponding confidence intervals
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




