function [ASTSE_models, ASTSE_params, ASTSE_w, state, chooses_min] = ASTSE_train( X_prev, Y_prev, S_prev, X_curr, Y_curr, S_curr, ...
                                                                                  ASTSE_models_prev, ASTSE_params_prev, ASTSE_w_prev, M, eta, lambda )
% ASTSE_TRAIN trains an Adaptive Skip-Train Structured Ensemble (AST-SE) at
% a certain timestep.
% 
% Requires:
%   X_prev            - input vectors from the previous timestep
%   Y_prev            - output values from the previous timestep
%   S_prev            - similarity matrix from the previous timestep
%   X_curr            - input vectors at the current timestep
%   Y_curr            - output values at the current timestep
%   S_curr            - similarity matrix at the current timestep
%   ASTSE_models_prev - unstructured predictors for AST-SE's components
%                       from the previous timestep
%   ASTSE_params_prev - parameters of AST-SE's components from the previous
%                       timestep
%   ASTSE_w_prev      - weights of AST-SE's components from the previous
%                       timestep
%   M                 - number of components within AST-SE
%   eta               - subsampling fraction
%   lambda            - nonnegative regularization parameter used to
%                       regularize the quadratic loss
% 
% Returns:
%   ASTSE_models      - selected or (re)trained unstructured predictors for
%                       AST-SE's components
%   ASTSE_params      - selected or learned components' parameters
%   ASTSE_w           - selected or learned components' weights
%   state             - selected AST-SE state
%   chooses_min       - a flag that equals either -1 or 1, indicating
%                       whether AST-SE's state was chosen directly or not,
%                       respectively

% Set the sampling fraction for the training graph to 80%
train_frac = 0.8;
% Initialize AST-SE's current state along with the direct/indirect state
% selection flag
state = -1;
chooses_min = -1;

% If the current timestep is the very first timestep, then train AST-SE as
% a weighted structured ensemble using the training graph from the current
% timestep since there is no data from previous timesteps
if X_prev == -1
    % Construct the training graph for the current timestep
    N_curr = size(X_curr,1);
    N_curr_train = round(train_frac*N_curr);
    X_curr_train = X_curr(1:N_curr_train, :);
    Y_curr_train = Y_curr(1:N_curr_train, :);
    S_curr_train = S_curr(1:N_curr_train, 1:N_curr_train);
    
    % Train an SE on the training graph from the current timestep
    [SE_models_curr, SE_params_curr] = SE_train(X_curr_train, Y_curr_train, S_curr_train, M, eta);
    % Determine the components' weights for the WSE using the training
    % graph from the current timestep
    preds_curr_train = SE_get_base_predictions(SE_models_curr, SE_params_curr, X_curr_train, S_curr_train);
    w_curr = get_weights(Y_curr_train, preds_curr_train, lambda);
    
    % Return the components of the trained WSE (i.e. the unstructured
    % predictors accompanied by their corresponding GCRF parameters) as
    % components for AST-SE, along with the components' weights
    ASTSE_models = SE_models_curr;
    ASTSE_params = SE_params_curr;
    ASTSE_w = w_curr;
    state = 1;
else
    % Get the validation graph from the previous timestep
    N_prev = size(X_prev,1);
    N_prev_train = round(train_frac*N_prev);
    X_prev_eval = X_prev(N_prev_train+1:end, :);
    Y_prev_eval = Y_prev(N_prev_train+1:end, :);
    S_prev_eval = S_prev(N_prev_train+1:end, N_prev_train+1:end);
    
    % Construct the training and validation graphs for the current timestep
    N_curr = size(X_curr,1);
    N_curr_train = round(train_frac*N_curr);
    X_curr_train = X_curr(1:N_curr_train, :);
    X_curr_eval = X_curr(N_curr_train+1:end, :);
    Y_curr_train = Y_curr(1:N_curr_train, :);
    Y_curr_eval = Y_curr(N_curr_train+1:end, :);
    S_curr_train = S_curr(1:N_curr_train, 1:N_curr_train);
    S_curr_eval = S_curr(N_curr_train+1:end, N_curr_train+1:end);
    
    % Create arrays for storing AST-SE's components (i.e. the unstructured
    % predictors accompanied by their corresponding GCRF parameters) and
    % their weights, from different AST-SE states
    all_ASTSE_models = cell(3,1);
    all_ASTSE_params = cell(3,1);
    all_ASTSE_w = cell(3,1);
    
    % Calculate the LOSS of:
    %   - the ensemble composed of:
    %      * the components from the previous timestep
    %      * the weights from the previous timestep
    % with respect to:
    %   - the validation graph from the previous timestep
    preds_prev_eval = SE_get_base_predictions(ASTSE_models_prev, ASTSE_params_prev, X_prev_eval, S_prev_eval);
    loss_0 = reg_quad_loss(ASTSE_w_prev, Y_prev_eval, preds_prev_eval, lambda);
    
    % Calculate the LOSS of:
    %   - the ensemble composed of:
    %      * the components from the previous timestep
    %      * the weights from the previous timestep
    % with respect to:
    %   - the validation graph from the current timestep
    preds_curr_eval = SE_get_base_predictions(ASTSE_models_prev, ASTSE_params_prev, X_curr_eval, S_curr_eval);
    loss_models_prev_w_prev = reg_quad_loss(ASTSE_w_prev, Y_curr_eval, preds_curr_eval, lambda);
    
    % Store components (unstructured predictors & GCRF parameters) and
    % weights for state 1
    all_ASTSE_models{1} = ASTSE_models_prev;
    all_ASTSE_params{1} = ASTSE_params_prev;
    all_ASTSE_w{1} = ASTSE_w_prev;
    
    % Determine whether state 1 should be selected
    if loss_models_prev_w_prev <= loss_0
        state = 1;
    else
        % Calculate the LOSS of:
	%   - the ensemble composed of:
	%      * the components from the previous timestep
	%      * weights learned on the training graph from the current
	%        timestep
	% with respect to:
	%   - the validation graph from the current timestep   
        preds_curr_train = SE_get_base_predictions(ASTSE_models_prev, ASTSE_params_prev, X_curr_train, S_curr_train);
        w_curr = get_weights(Y_curr_train, preds_curr_train, lambda);
        loss_models_prev_w_curr = reg_quad_loss(w_curr, Y_curr_eval, preds_curr_eval, lambda);
        
        % Store components (unstructured predictors & GCRF parameters) and
        % weights for state 2
        all_ASTSE_models{2} = ASTSE_models_prev;
        all_ASTSE_params{2} = ASTSE_params_prev;
        all_ASTSE_w{2} = w_curr;
        
        % Determine whether state 2 should be selected
        if loss_models_prev_w_curr <= loss_0
            state = 2;
        else
	    % Calculate the LOSS of:
	    %   - the ensemble composed of:
	    %      * M_selected components from the previous timestep
	    %      * M-M_selected components retrained on the training
	    %        graph from the current timestep
	    %      * weights learned on the training graph from the current
	    %        timestep
	    % with respect to:
	    %   - the validation graph from the current timestep
            sorted_weights = sortrows([(1:M)', w_curr'], 2);
            sorted_weight_diffs = (sorted_weights(2:M,2) - sorted_weights(1:M-1,2));
            [~,selection_threshold] = max(sorted_weight_diffs);
            top_weights_indices = sorted_weights(selection_threshold+1:end,1);
            M_selected = size(top_weights_indices,1);
            
            SE_models_comb = cell(M,1);
            SE_params_comb = zeros(M,2);
            for m = 1 : M_selected
                SE_models_comb{m} = ASTSE_models_prev{top_weights_indices(m)};
                SE_params_comb(m,:) = ASTSE_params_prev(top_weights_indices(m),:);
            end

            [SE_models_retrained, SE_params_retrained] = SE_train(X_curr_train, Y_curr_train, S_curr_train, M - M_selected, eta);
            for m = M_selected + 1 : M
                SE_models_comb{m} = SE_models_retrained{m - M_selected};
                SE_params_comb(m,:) = SE_params_retrained(m - M_selected,:);
            end
            
            preds_curr_train = SE_get_base_predictions(SE_models_comb, SE_params_comb, X_curr_train, S_curr_train);
            w_curr = get_weights(Y_curr_train, preds_curr_train, lambda);
            preds_curr_eval = SE_get_base_predictions(SE_models_comb, SE_params_comb, X_curr_eval, S_curr_eval);
            loss_models_comb_w_curr = reg_quad_loss(w_curr, Y_curr_eval, preds_curr_eval, lambda);
            
            % Store components (unstructured predictors & GCRF parameters)
            % and weights for state 3
            all_ASTSE_models{3} = SE_models_comb;
            all_ASTSE_params{3} = SE_params_comb;
            all_ASTSE_w{3} = w_curr;
            
            % Determine whether state 3 should be selected
            if loss_models_comb_w_curr <= loss_0
                state = 3;
            else
		% Choose the state in which the minimum loss was obtained
                [~,state] = min( [loss_models_prev_w_prev, loss_models_prev_w_curr, loss_models_comb_w_curr] );                
                chooses_min = 1;
            end
        end
    end
    
    % Return the final components (unstructured predictors & GCRF
    % parameters) for AST-SE, along with the components' weights
    ASTSE_models = all_ASTSE_models{state};
    ASTSE_params = all_ASTSE_params{state};
    ASTSE_w = all_ASTSE_w{state};
end

end
