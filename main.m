clc; close all; clear all;

 
load('Data/H3N2_data.mat')
T = size(XList,1);



M = 30;
eta = 0.3;
lambda = 0.1;



ERRORS = [];

TSE_models_PREV = -1;
TSE_params_PREV = -1;
w_PREV = -1;


rng(1, 'twister');

for t = 1 : T - 1
    

    
    % Construct train and test data
    Xtrain = XList{t};
    Ytrain = YList{t};
    Strain = SSS;
    Ntrain = size(Xtrain,1);
    
    Xtest = XList{t+1};
    Ytest = YList{t+1};
    Stest = SSS;
    Ntest = size(Xtest,1);
    
    
    
    % Unstructured regressor
    base_model = base_train(Xtrain, Ytrain);
    Rtrain = base_predict(base_model,Xtrain);
    Rtest = base_predict(base_model,Xtest);
    UR_testMSE = calc_MSE(Ytest, Rtest);
    
    
    
    % GCRF
    theta = GCRF_train(Ytrain,Strain,Rtrain);
    mu = GCRF_predict(theta, 1, Stest, Rtest);
    GCRF_testMSE = calc_MSE(Ytest, mu);
    
    
    
    % TSE
    [TSE_models, TSE_params] = TSE_train(Xtrain, Ytrain, Strain, M, eta);
    predictions_test = TRE_get_predictions(TSE_models, TSE_params, Xtest, Stest);
    avg_mu = mean(predictions_test,2);
    TSE_testMSE = calc_MSE(Ytest, avg_mu);
    
    
    
    % Weighted TSE
    predictions_train = TRE_get_predictions(TSE_models, TSE_params, Xtrain, Strain);
    w = infer_weights(Ytrain, predictions_train, lambda);
    wavg_mu = predictions_test * w';
    WTSE_testMSE = calc_MSE(Ytest, wavg_mu);
    
    
    
    % Skip-step TSE
    if t == 1
        X_PREV = -1;
        Y_PREV = -1;
        S_PREV = -1;
    else
        X_PREV = XList{t-1};
        Y_PREV = YList{t-1};
        S_PREV = SSS;
    end
    [TSE_models_FINAL, TSE_params_FINAL, w_FINAL, state, chooses_min] = SSTSE_train(X_PREV, Y_PREV, S_PREV, XList{t}, YList{t}, SSS, ...
                                                                                    TSE_models_PREV, TSE_params_PREV, w_PREV, M, eta, lambda);

    TSE_models_PREV = TSE_models_FINAL;
    TSE_params_PREV = TSE_params_FINAL;
    w_PREV = w_FINAL;


    predictions_test = TRE_get_predictions(TSE_models_FINAL, TSE_params_FINAL, Xtest, Stest);
    sswavg_mu = predictions_test * w_FINAL';
    SSWTSE_testMSE = calc_MSE(Ytest, sswavg_mu);
    
    
    
    sorted_errors = sortrows([[1:5]',[UR_testMSE, GCRF_testMSE, TSE_testMSE, WTSE_testMSE, SSWTSE_testMSE]'],2);
    SSWTSE_rank = find(sorted_errors(:,1)==5);
    fprintf('%d(%d)\t|\t', state, chooses_min);
    
    
    % Errors
    ERRORS = [ERRORS ; [UR_testMSE, GCRF_testMSE, TSE_testMSE, WTSE_testMSE, SSWTSE_testMSE]];
    fprintf('(%d)\t%f\t%f\t%f\t%f\t%f\t%d\t|...\n', t, UR_testMSE, GCRF_testMSE, TSE_testMSE, WTSE_testMSE, SSWTSE_testMSE, SSWTSE_rank);
       
end

fprintf('\n\n');

fprintf('ERRORS:\n');
for col = 1 : size(ERRORS,2)
    mean_diff_pair = calc_conf_interval(ERRORS(:,col));
    fprintf('%f\t%f\n', mean_diff_pair(1), mean_diff_pair(2));
end





