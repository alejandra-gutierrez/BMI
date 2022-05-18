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
  t_mvt = 210;
  t_start = 1;
  t_pre_mvt = 300;
  
  spike_rates_test = get_spike_rates2(test_data, windowsize, t_step, t_start); % time limiting step
    % this is a cell array of size [N_trials x 1]
    % containing [N_neurons x t_max_each] spike rates
  
  

  % ... compute position at the given timestep.
  for m=1:N_trials_test
    pos0 = test_data(m).startHandPos;   % [x; y]
    test_data(m).decodedHandPos = [];
    t_end = size(test_data(m).spikes, 2);
    
    % STEP 1: COMPUTE PREDICTED DIRECTION
    sr = sum(test_data(m).spikes(:, 1:t_pre_mvt), 2)'; % [1 x N_neurons]
    dir = predict(modelParameters(1).knn, sr); % model same for all entries
       
    
    % STEP 2: COMPUTE CURRENT POSITION 
    V_red = modelParameters(dir).V_red;
    M = modelParameters(dir).M;
    spikes_mean = mean(spike_rates_test{m}, 2);
    principal_sr_test = V_red'*(spike_rates_test{m} - spikes_mean);

   
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

    x = 0; y=0;
    
    x(m) = sum(vel_x_estimated(t_mvt:end)) + test_data(m).startHandPos(1);
    y(m) = sum(vel_y_estimated(t_mvt:end)) + test_data(m).startHandPos(2);
  end

   
end


function spike_rates = get_spike_rates2(trials, windowsize, t_step, t_start)
    % output size spike_rates: cell [N_trials x N_angles]
%       each is a double, size [N_neurons  x (t_max_each/t_step)]

% input trials - trial(n, k), spikes size(N_neurons, t_max)
%input t_step: not there = assume keep all time steps
% if t_step == 0 -> assume want default size reduction = windowsize/2


    [N_trials, N_angles]= size(trials);
    N_neurons = size(trials(1,1).spikes,1);
    
    
    if ~exist('t_step', 'var') || isempty(t_step)
        t_step = 1;
    elseif t_step <=0
        t_step = ceil(windowsize/2);
    end
    if ~exist('t_start', 'var') || isempty(t_start)
        t_start = 1;
    end
    
    t_start = t_start - 1; % offset for iteration
    
    % make sure these are integers!
    t_start = floor(max([t_start, 0]));
    windowsize = ceil(windowsize) ;
    t_step = ceil(max([t_step, 1]));

    spike_rates = cell(N_trials, N_angles);
    
    max_t = 0;

    for n = 1:N_trials
        for k = 1:N_angles
            spikes = trials(n,k).spikes;
            timesteps = size(spikes, 2);
            spike_rates{n, k} = zeros(N_neurons, ceil(timesteps/t_step));
            
            if timesteps>max_t
                max_t = timesteps;
            end
            
            for t = t_start+windowsize:t_step:timesteps
                rate = sum(spikes(:, t-windowsize+1:t), 2)/windowsize*1000;
                spike_rates{n, k}(:, ceil((t-t_start)/t_step)) = rate;

            end
        end
    end
    
end

