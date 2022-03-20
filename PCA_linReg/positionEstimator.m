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
  windowsize = 26; % time window for velocity and spike rate estimation
  t_mvt = 290; % hand movement start
    
  spike_rates_test = get_spike_rates2(test_data, windowsize);
    % this is a cell array of size [N_trials x 1]
    % containing [N_neurons x t_max] spike rates
  
  model_knn = modelParameters(9).knn;
  model_KNNspikesr = modelParameters(9).KNNSpikesr;
  model_KNNlabels = modelParameters(9).KNNLabels;
  n_neighbours = modelParameters(9).n_neighbours;
  N_KNN = size(model_KNNspikesr, 1);    %[N_trials*N_angles x N_neurons]


  % ... compute position at the given timestep.
  for m=1:N_trials_test
    pos0 = test_data(m).startHandPos;   % [x; y]
    test_data(m).decodedHandPos = [];
    
    % STEP 1: COMPUTE PREDICTED DIRECTION

    % dir = knn_pred(spike_rates_test{m}, knn_model)
    % dir = knn_pred(test_data, knn_model); % sth similar
    sr = sum(test_data(m).spikes(:, 1:t_mvt), 2)'; % [1 x N_neurons]
%     dir = predict(model_knn, sr); % currently using toolbox
    
    dist = zeros(1, size(model_KNNspikesr, 1)); 
    for n_it = 1:N_KNN
        dist(n_it) = sqrt(sum((sr - model_KNNspikesr(n_it, :)).^2));
    end
    [sortedDist, ind] = sort(dist, 2);
    nearest_neighbours = ind(2:n_neighbours+1); % gives index in long list of which neighbour is close
    
    % gives the direction corresponding to this neighbour (from the label)
    nearest_dirs(1:n_neighbours) = model_KNNlabels(nearest_neighbours); 
    dir = mode(nearest_dirs(1:n_neighbours));  % find the most common value in the neighbourhood

    fprintf("\nPredicted dir: %g\n", dir);
    
    %dir = 9; % non-specific direction for now
    
    % STEP 2: COMPUTE CURRENT POSITION 
    V_red = modelParameters(dir).V_red;
    M = modelParameters(dir).M;
    wX = modelParameters(dir).PCAweightsX;
    wY = modelParameters(dir).PCAweightsY;
%     fprintf("Model Parameters: M=%g,  size2 V_red=%g, size wX=[%g,%g]\n", M, size(V_red,2), size(wX,1), size(wX, 2));
    spikes_mean = mean(spike_rates_test{m}, 2);
    principal_sr_test = V_red'*(spike_rates_test{m} - spikes_mean);
    
    velx_estimated = wX'*principal_sr_test;
    vely_estimated = wY'*principal_sr_test;
    x(m) = sum(velx_estimated(t_mvt:end)) + test_data(m).startHandPos(1);
    test_data(m).decodedHandPos(1) =x;
    y(m) = sum(vely_estimated(t_mvt:end)) + test_data(m).startHandPos(2);
    test_data(m).decodedHandPos(2) = y;

  end

  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
   
end