%%% Team Members: WRITE YOUR TEAM MEMBERS' NAMES HERE
%%% BMI Spring 2015 (Update 17th March 2015)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         PLEASE READ BELOW            %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function positionEstimator has to return the x and y coordinates of the
% monkey's hand position for each trial using only data up to that moment
% in time.
% You are free to use the whole trials for training the classifier.

% To evaluate performance we require from you two functions:

% A training function named "positionEstimatorTraining" which takes as
% input the entire (not subsampled) training data set and which returns a
% structure containing the parameters for the positionEstimator function:
% function modelParameters = positionEstimatorTraining(training_data)
% A predictor named "positionEstimator" which takes as input the data
% starting at 1ms and UP TO the timepoint at which you are asked to
% decode the hand position and the model parameters given by your training
% function:

% function [x y] = postitionEstimator(test_data, modelParameters)
% This function will be called iteratively starting with the neuronal data 
% going from 1 to 320 ms, then up to 340ms, 360ms, etc. until 100ms before 
% the end of trial.


% Place the positionEstimator.m and positionEstimatorTraining.m into a
% folder that is named with your official team name.

% Make sure that the output contains only the x and y coordinates of the
% monkey's hand.


function [modelParameters] = positionEstimatorTraining(training_data)
% Arguments:

% - training_data:
%     training_data(n,k)              (n = trial id,  k = reaching angle)
%     training_data(n,k).trialId      unique number of the trial
%     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
%     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)

% ... train your model

% Return Value:

% - modelParameters:
%     single structure containing all the learned parameters of your
%     model and which can be used by the "positionEstimator" function.

t_start = tic;
tic;
[N_trials_tr, N_angles] = size(training_data);
N_neurons = size(training_data(1).spikes, 1);

windowsize = 20;
t_mvt = 210;
t_pre_mvt = 300;
t_step = windowsize/2;
t_step = ceil(t_step);
n_neighbours = 12;
proportion = 2/100; % th for selection of principal components

fprintf("\nFinding spike rates and velocities...");
[velx_tr, vely_tr, ~] = getvel2(training_data, windowsize, t_step, t_mvt);
spike_rate = get_spike_rates2(training_data, windowsize, t_step, t_mvt);
fprintf("Spike_Rate done...\n");
toc;

%% TRAIN KNN MODEL

fprintf("Training KNN model...");
spikesr = zeros(N_angles*N_trials_tr, N_neurons);
labels = zeros(1, N_angles*N_trials_tr);
for k_it = 1:N_angles
    for n_it = 1:N_trials_tr
         spikesr( (k_it-1)*N_trials_tr + n_it, :) = sum(training_data(n_it, k_it).spikes(:, 1:t_pre_mvt), 2)';           
        labels( (k_it-1)*N_trials_tr + n_it) = k_it;
    end
end

% knn = fitcknn(spikesr, labels);
for k_it = 1:N_angles+1
    modelParameters(k_it).KNNSpikesr = spikesr;
    modelParameters(k_it).KNNLabels = labels;
    % modelParameters(k_it).knn = knn;
    modelParameters(k_it).n_neighbours = n_neighbours;
end
fprintf("KNN model done. "); toc;


%% TRAIN POSITION ESTIMATOR
fprintf("Extracting Principal component vectors from data...");
%   spike_rate_av_trials = make_av_spike_rate(spike_rate);
%   [principal_spikes, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, 0.05);
%   principal_spikes_tr = cell(N_trials_tr, N_angles+1);
% 
%   modelParameters(9).M = M;
%   modelParameters(9).dir = 0;
%   modelParameters(9).Vs = Vs;
%   modelParameters(9).Ds = Ds;
%   modelParameters(9).V_red = Vs(:,1:M);


for k_it =0:N_angles
    spike_rate_av_trials = make_av_spike_rate(spike_rate, k_it);
    [~, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, proportion);
    dir = k_it;
    if k_it == 0
        k_it = N_angles+1;
    end
    V_red = Vs(:, 1:M);
    modelParameters(k_it).M = M; % keep same M for all
    modelParameters(k_it).dir = dir;
    modelParameters(k_it).Vs = Vs;
    modelParameters(k_it).Ds = Ds;
    modelParameters(k_it).V_red = V_red;

    for n_it = 1:N_trials_tr
        if k_it == N_angles+1
            for k = 1:N_angles
                % make a specific array for non-specific training
                spikes_mean = mean(spike_rate{n_it, k}, 2);
                principal_spikes_0{n_it, k} = V_red'*(spike_rate{n_it, k}-spikes_mean);
            end
        else
           spikes_mean = mean(spike_rate{n_it, k_it}, 2);
           principal_spikes_tr{n_it, k_it} = V_red'*(spike_rate{n_it, k_it}-spikes_mean);
        end
        fprintf(".");
    end
    fprintf("\n"); 
end
fprintf("Extracted PCA parameters.\n"); toc;

fprintf("Starting Linear Regression.\t");

for k_it = 0:N_angles
    fprintf("k=%g.\t", k_it);
    if (k_it ==0) % non-direction specific training
        PCA_components_weights_x = linearRegression2(principal_spikes_0, velx_tr, k_it);
        PCA_components_weights_y = linearRegression2(principal_spikes_0, vely_tr, k_it);
        k_it = N_angles+1;
    else  % direction specific training
        PCA_components_weights_x = linearRegression2(principal_spikes_tr, velx_tr, k_it);
        PCA_components_weights_y = linearRegression2(principal_spikes_tr, vely_tr, k_it);
    end
    
    modelParameters(k_it).PCAweightsX = PCA_components_weights_x;
    modelParameters(k_it).PCAweightsY = PCA_components_weights_y;
end
fprintf("\n Done.\n");
fprintf("Model Parameters:\n");
% print model parameters
for k_it = 1:N_angles+1
    M = modelParameters(k_it).M;
    dir = modelParameters(k_it).dir;
    V_red = modelParameters(k_it).V_red;
    Vs = modelParameters(k_it).Vs;
    Ds = modelParameters(k_it).Ds;
    wX = modelParameters(k_it).PCAweightsX;
    wY = modelParameters(k_it).PCAweightsY;
    fprintf("dir=%g, M=%g,  size V_red=[%g, %g], size wX=[%g,%g], size wY=[%g,%g]\n",...
    dir, M, size(V_red,1),size(V_red,2), size(wX,1), size(wX, 2), size(wY,1), size(wY, 2));
    
end
fprintf("\nFinished Training.\n");
toc; fprintf("\n");
end



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
  windowsize = 35; % time window for velocity and spike rate estimation
  t_step = windowsize /2.5;
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
    
    V_red = modelParameters(dir).V_red;
    M = modelParameters(dir).M;
    wX = modelParameters(dir).PCAweightsX;
    wY = modelParameters(dir).PCAweightsY;
%     fprintf("Model Parameters: M=%g,  size2 V_red=%g, size wX=[%g,%g]\n", M, size(V_red,2), size(wX,1), size(wX, 2));
    spikes_mean = mean(spike_rates_test{m}, 2);
    principal_sr_test = V_red'*(spike_rates_test{m} - spikes_mean);
    
    
    t_red = t_start:t_step:t_end;
    t_shift = [2:length(t_red), length(t_red)];
    
    L_pr = zeros(M, t_end);
    
    for it = 0:t_step-1
        lin_elmt = (principal_sr_test(:, t_shift) - principal_sr_test)*it/t_step;
        L_pr(:, t_red + it) = principal_sr_test + lin_elmt;  % linear interpolation
    end
%     for it = 0:t_step-1
%         L_pr(:, t_red + it) = principal_sr_test; % very rough
%         interpolation
%     end
    L_pr(:,t_end+1:end) = []; % remove possibly extra values
    
%     figure(1); hold off;
%     plot(t_red, principal_sr_test, 'o'); hold on
%     plot(1:t_end, L_pr);
    
    
%     velx_estimated = wX'*principal_sr_test;
%     vely_estimated = wY'*principal_sr_test;
    velx_estimated = wX'*L_pr; 
    vely_estimated = wY'*L_pr;
    
    x = 0; y=0;
    x(m) = sum(velx_estimated(t_mvt:end)) + test_data(m).startHandPos(1);
    test_data(m).decodedHandPos(1) = x(m);
    y(m) = sum(vely_estimated(t_mvt:end)) + test_data(m).startHandPos(2);
    test_data(m).decodedHandPos(2) = y(m);

  end

  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
   
end