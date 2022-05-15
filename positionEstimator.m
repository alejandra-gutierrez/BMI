function [x, y] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
  test_data = test_data(:); % make sure test_data is linear for testing
  N_trials_test = size(test_data, 1);
  t_max = size(test_data(1).spikes, 2); % how long is the current run
  
  % hardcoded parameters
  windowsize = 15; % time window for velocity and spike rate estimation
  t_step = windowsize /2;
  t_step = ceil(t_step); % prevent weirdness and unpredicatability
  t_mvt = 290; % hand movement start
  t_start = 1;
  
  spike_rates_test = get_spike_rates2(test_data, windowsize, t_step, t_start); % time limiting step
    % this is a cell array of size [N_trials x 1]
    % containing [N_neurons x t_max_each] spike rates
  
  %model_knn = modelParameters(9).knn;
  model_KNNspikesr = modelParameters(9).KNNSpikesr;
  model_KNNlabels = modelParameters(9).KNNLabels;
  n_neighbours = modelParameters(9).n_neighbours;
  N_KNN = size(model_KNNspikesr, 1);    %[N_trials*N_angles x N_neurons]


  % ... compute position at the given timestep.
  for m=1:N_trials_test
    pos0 = test_data(m).startHandPos;   % [x; y]
    test_data(m).decodedHandPos = [];
    t_end = size(test_data(m).spikes, 2);
    
    % STEP 1: COMPUTE PREDICTED DIRECTION
    sr = sum(test_data(m).spikes(:, 1:t_mvt), 2)'; % [1 x N_neurons]
    
%     dir = predict(model_knn, sr); % using toolbox
    
    dist = zeros(1, size(model_KNNspikesr, 1)); 
    for n_it = 1:N_KNN
        dist(n_it) = sqrt(sum((sr - model_KNNspikesr(n_it, :)).^2));
    end
    [sortedDist, ind] = sort(dist, 2);
    nearest_neighbours = ind(2:n_neighbours+1); % gives index in long list of which neighbour is close
    
    % gives the direction corresponding to this neighbour (from the label)
    nearest_dirs(1:n_neighbours) = model_KNNlabels(nearest_neighbours); 
    dir = mode(nearest_dirs(1:n_neighbours));  % find the most common value in the neighbourhood

%     fprintf("\nPredicted dir: %g\n", dir);
    
    
    % STEP 2: COMPUTE CURRENT POSITION 
%     A = modelParameters(dir).A; % calculated in training
%     W = modelParameters(dir).W; % calculated in training
%     H = modelParameters(dir).H; % calculated in training
%     Q = modelParameters(dir).Q; % calculated in training


%     P = modelParameters(dir).P;
    V_red = modelParameters(dir).V_red;
    M = modelParameters(dir).M;
    spikes_mean = mean(spike_rates_test{m}, 2);
    principal_sr_test = V_red'*(spike_rates_test{m} - spikes_mean);
%     principal_sr_test = (spike_rates_test{m} - spikes_mean);

    
    
    t_red = t_start:t_step:t_end;
    t_shift = [2:length(t_red), length(t_red)];
    L_pr = zeros(M, t_end);
    
    for it = 0:t_step-1
        lin_elmt = (principal_sr_test(:, t_shift) - principal_sr_test)*it/t_step;
        L_pr(:, t_red + it) = principal_sr_test + lin_elmt;  % linear interpolation
    end

    L_pr(:,t_end+1:end) = []; % remove possibly extra values
    
    vel_x_estimated = predict(modelParameters(dir).x, L_pr, 'ObservationsIn', 'columns');
    vel_y_estimated = predict(modelParameters(dir).y, L_pr, 'ObservationsIn', 'columns');

%     % KALMAN: BAYESIAN RECURSIVE
%     estimated_params(:,1) = [0; 0];
% %     pred_v = (A*x_k_minus_1);   
%     pred_params(:,1) = A*[0; 0];
%     j = 2;
%     for k = t_mvt+1:1:t_end
%             pred_params(:,j) = A*estimated_params(:,j-1);
%             pred_P = A*P*A' + W;
%             K = pred_P*H'/(H*pred_P*H' + Q);
%             estimated_params(:,j) = pred_params(:,j) + K*(L_pr(:,j) -(H*pred_params(:,j)));
%             j = j + 1;
%             P = (eye(size(A,2))-(K*H))*pred_P;    
% 
%     end
% 
%     modelParameters(dir).P = P;
% 
%     velx_estimated = estimated_params(1,:);
%     vely_estimated = estimated_params(2,:);
% 
% 
    x = 0; y=0;
% 
%     x2(m) = sum(velx_estimated(t_mvt:end)) + test_data(m).startHandPos(1);
%     y2(m) = sum(vely_estimated(t_mvt:end)) + test_data(m).startHandPos(2);
% 
% 
%     test_data(m).decodedHandPos(1) = x2(m);
%     x(m) = x2(m);
%     test_data(m).decodedHandPos(2) = y2(m);
%     y(m) = y2(m);
% 
% 
%     newModelParameters = modelParameters;

      x(m) = sum(vel_x_estimated(t_mvt:end)) + test_data(m).startHandPos(1);
      y(m) = sum(vel_y_estimated(t_mvt:end)) + test_data(m).startHandPos(2);
  end

  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
   
end