function [TSE_models_FINAL, TSE_params_FINAL, w_FINAL, state, chooses_min] = SSTSE_train( X_PREV, Y_PREV, S_PREV, X_CURR, Y_CURR, S_CURR, ...
                                                                                                     TSE_models_PREV, TSE_params_PREV, w_PREV, M, eta, lambda )
                                                                              
train_frac = 0.8;
loss_PP_PREVEval = 1000;
loss_PP_CURREval = 1000;
loss_PC_CURREval = 1000;
loss_PCC_CURREval = 1000;
loss_CC_CURREval = 1000;
state = -1;
chooses_min = 0;

if size(X_PREV,1) == 1 && X_PREV == -1
    
    N_CURR = size(X_CURR,1);

    N_CURRTrain = round(train_frac*N_CURR);
    X_CURRTrain = X_CURR(1:N_CURRTrain, :);
    %X_CURREval = X_CURR(N_CURRTrain+1:end, :);
    Y_CURRTrain = Y_CURR(1:N_CURRTrain, :);
    %Y_CURREval = Y_CURR(N_CURRTrain+1:end, :);
    S_CURRTrain = S_CURR(1:N_CURRTrain, 1:N_CURRTrain);
    %S_CURREval = S_CURR(N_CURRTrain+1:end, N_CURRTrain+1:end);
    
    [TSE_models, TSE_params] = TSE_train(X_CURRTrain, Y_CURRTrain, S_CURRTrain, M, eta);
    predictions_C_CURRTrain = TRE_get_predictions(TSE_models, TSE_params, X_CURRTrain, S_CURRTrain);
    w = infer_weights(Y_CURRTrain, predictions_C_CURRTrain, lambda);
    
    TSE_models_FINAL = TSE_models;
    TSE_params_FINAL = TSE_params;
    w_FINAL = w;
    state = 1;
    
else

    N_PREV = size(X_PREV,1);
    
    N_PREVTrain = round(train_frac*N_PREV);
    %X_PREVTrain = X_PREV(1:N_PREVTrain, :);
    X_PREVEval = X_PREV(N_PREVTrain+1:end, :);
    %Y_PREVTrain = Y_PREV(1:N_PREVTrain, :);
    Y_PREVEval = Y_PREV(N_PREVTrain+1:end, :);
    %S_PREVTrain = S_PREV(1:N_PREVTrain, 1:N_PREVTrain);
    S_PREVEval = S_PREV(N_PREVTrain+1:end, N_PREVTrain+1:end);






    N_CURR = size(X_CURR,1);
    
    N_CURRTrain = round(train_frac*N_CURR);
    X_CURRTrain = X_CURR(1:N_CURRTrain, :);
    X_CURREval = X_CURR(N_CURRTrain+1:end, :);
    Y_CURRTrain = Y_CURR(1:N_CURRTrain, :);
    Y_CURREval = Y_CURR(N_CURRTrain+1:end, :);
    S_CURRTrain = S_CURR(1:N_CURRTrain, 1:N_CURRTrain);
    S_CURREval = S_CURR(N_CURRTrain+1:end, N_CURRTrain+1:end);


    
    
    
    ALL_TSE_models = cell(3,1);
    ALL_TSE_params = cell(3,1);
    ALL_w = cell(3,1);
    
    

    % LOSS_PP (PREVEval)
    [predictions_P_PREVEval] = TRE_get_predictions(TSE_models_PREV, TSE_params_PREV, X_PREVEval, S_PREVEval);
    loss_PP_PREVEval = WM_objective(w_PREV, Y_PREVEval, predictions_P_PREVEval, lambda);

    % LOSS_PP (CURREval)
    [predictions_P_CURREval] = TRE_get_predictions(TSE_models_PREV, TSE_params_PREV, X_CURREval, S_CURREval);
	loss_PP_CURREval = WM_objective(w_PREV, Y_CURREval, predictions_P_CURREval, lambda);
	
    ALL_TSE_models{1} = TSE_models_PREV;
    ALL_TSE_params{1} = TSE_params_PREV;
    ALL_w{1} = w_PREV;
    
    if loss_PP_CURREval <= loss_PP_PREVEval
        state = 1;
    else
        
        % LOSS_PC (CURREval)
        [predictions_P_CURRTrain] = TRE_get_predictions(TSE_models_PREV, TSE_params_PREV, X_CURRTrain, S_CURRTrain);
        w_CURR = infer_weights(Y_CURRTrain, predictions_P_CURRTrain, lambda);
        loss_PC_CURREval = WM_objective(w_CURR, Y_CURREval, predictions_P_CURREval, lambda);
        
        ALL_TSE_models{2} = TSE_models_PREV;
        ALL_TSE_params{2} = TSE_params_PREV;
        ALL_w{2} = w_CURR;
        
        if loss_PC_CURREval <= loss_PP_PREVEval
            state = 2;
        else
            
            % LOSS_[P,C]C (CURREval)
            sorted_weights = sortrows([[1:M]' , w_CURR'],2);
            sorted_weights_diffs = (sorted_weights(2:M,2) - sorted_weights(1:M-1,2));
            [~,max_index] = max(sorted_weights_diffs);
            chosen_indices = sorted_weights(max_index+1:end,1);

            M_PREV = size(chosen_indices,1);

            TSE_models_BOTH = cell(M,1);
            TSE_params_BOTH = zeros(M,2);
            for m = 1 : M_PREV
                TSE_models_BOTH{m} = TSE_models_PREV{chosen_indices(m)};
                TSE_params_BOTH(m,:) = TSE_params_PREV(chosen_indices(m),:);
            end

            [TSE_models_BOTH_2, TSE_params_BOTH_2] = TSE_train(X_CURRTrain, Y_CURRTrain, S_CURRTrain, M - M_PREV, eta);

            for m = M_PREV + 1 : M
                TSE_models_BOTH{m} = TSE_models_BOTH_2{m - M_PREV};
                TSE_params_BOTH(m,:) = TSE_params_BOTH_2(m - M_PREV,:);
            end

            predictions_PC_CURRTrain = TRE_get_predictions(TSE_models_BOTH, TSE_params_BOTH, X_CURRTrain, S_CURRTrain);
            w_CURR = infer_weights(Y_CURRTrain, predictions_PC_CURRTrain, lambda);
            predictions_PC_CURREval = TRE_get_predictions(TSE_models_BOTH, TSE_params_BOTH, X_CURREval, S_CURREval);
            loss_PCC_CURREval = WM_objective(w_CURR, Y_CURREval, predictions_PC_CURREval, lambda);

            ALL_TSE_models{3} = TSE_models_BOTH;
            ALL_TSE_params{3} = TSE_params_BOTH;
            ALL_w{3} = w_CURR;
            
            if loss_PCC_CURREval <= loss_PP_PREVEval
                state = 3;
            else
                [~,state] = min( [loss_PP_CURREval, loss_PC_CURREval, loss_PCC_CURREval] );                
                chooses_min = 1;
            end
        end
    end
    
    
    % Return the final models + params + weighs
    TSE_models_FINAL = ALL_TSE_models{state};
    TSE_params_FINAL = ALL_TSE_params{state};
    w_FINAL = ALL_w{state};
    
end



end

