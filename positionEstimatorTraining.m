function [modelParameters] = positionEstimatorTraining(training_data)
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
 

% timeLag = 75; % time lag between neural signals and corresponding movement 

% training_input should be Nx1 cell array, where N is number of
% observations. 
% Number of observations is number of directions (8) multiplied by number
% of trials used for training data (80).
% Each cell contains c x s matrix, where c is the number of features, and s
% is the time duration of the sequence.
validation_trials = 10;
N_trials = size(training_data, 1);
N_directions = size(training_data, 2);
N = N_trials * N_directions;

windowsize = 20; % 20 ms data windows
modelParameters.windowsize = windowsize;

training_input = cell(N,1); % initialise predictor array (X)
training_output = cell(N,1); % initialise response array (Y)

% input c x s matrix (spike data) into input: raw, unbinned (98 x duration)
% input r x s matrix (position data) into output: raw, unbinned (2 x
% duration)
for n = 1:N
    current_row = mod(n-1, N_trials)+1;
    current_column = fix((n - 1)/N_trials) + 1;
    training_input{n} = training_data(current_row, current_column).spikes;
    training_output{n} = training_data(current_row, current_column).handPos(1:2,:);

end

%% TRAIN KNN MODEL

N_neurons = size(training_input{1}, 1);
fprintf("Training KNN model...");
spikesr = zeros(N, N_neurons);
labels = zeros(1, N);
t_pre_mvt = 300;
modelParameters.t_pre_mvt = t_pre_mvt;

for n = 1:N
    spikesr(n, :) = sum(training_input{n}(:, 1:t_pre_mvt), 2)';           
    labels(n) = fix((n - 1)/N_trials) + 1;
end
knn = fitcknn(spikesr, labels, 'NumNeighbors',21);
modelParameters.KNNSpikesr = spikesr;
modelParameters.KNNLabels = labels;
modelParameters.knn = knn;
modelParameters.n_neighbours = 21;
fprintf("KNN model done. "); toc;
%%
t_mvt = 210; % start of relevant neural data for movement 

% t_mvt should be amultiple of windowsize to not upset binning
if mod(t_mvt, windowsize) ~= 0
    t_mvt = t_mvt - mod(t_mvt, windowsize);
end
modelParameters.t_mvt = t_mvt;
% alter sizing for binning the data. If the duration is not a multiple of
% windowsize, the extra data points are clipped off (these will be part of
% the final 100 data points and therefore redundant to position prediction.
% The same process is applied to both input and output training sets.
for n = 1:N
    clip = mod(size(training_input{n}, 2),windowsize);
    t = training_input{n};
    training_input{n} = [];
    training_input{n} = t(:,t_mvt+1:end-clip);
    t = training_output{n};
    training_output{n} = [];
    training_output{n} = t(:,t_mvt+1:end-clip);
end

% convert the output position data into velocity, and bin into chunks of
% size windowsize. Velocity of each bin calculated by subtracting first
% position value in bin from last in bin, and dividing by windowsize.
training_velocities = cellfun(@(x) downsize_and_convert_to_velocity(x, windowsize), training_output, 'un', 0);

% bin the spike data. spikes are summed in each bin, and converted to a
% rate in Hz. 
training_input = cellfun(@(x) (squeeze(sum(reshape(x,size(x,1),windowsize,[]),2))/windowsize)*1000, training_input, 'un', 0);
%% TRAIN POSITION ESTIMATOR
fprintf("Extracting Principal component vectors from data...");
 
%% PCA
proportion = 2/100;
direction_specific_data = reshape(training_input, N_trials, N_directions);
    t = cellfun(@(x) any(isnan(x(:))), direction_specific_data, 'UniformOutput', false);
    s= sum([t{:}]);
for k_it = 1:N_directions
    spike_rate_av_trials = make_av_spike_rate(training_input, N_trials, k_it);
    [Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, proportion);
    dir = k_it;
%     sum(any(isnan(spike_rate_av_trials)))
    V_red = Vs(:, 1:M);
    modelParameters.M{k_it} = M; % keep same M for all
    modelParameters.dir{k_it} = dir;
    modelParameters.Vs{k_it} = Vs;
    modelParameters.Ds{k_it} = Ds;
    modelParameters.V_red{k_it} = V_red;
 
    
    
    for n = 1:N_trials
        principle_spikes{n, k_it} = V_red'*direction_specific_data{n, k_it};
    end
    mu(:,k_it) = mean([principle_spikes{:,k_it}],2); % mean spike rate for each neuron across all trials for each direction
    sig(:, k_it) = std([principle_spikes{:,k_it}],0,2); % standard deviation of spike rate for each neuron across all trials for each direction
    % normalise
    for n = 1:N_trials
%         direction_specific_data{n,k_it} = (direction_specific_data{n, k_it}-mu(:,k_it))./sig(:, k_it);
        principle_spikes{n,k_it} = (principle_spikes{n, k_it})./sig(:,k_it);

    end

end

%% NON DIRECTION SPECIFIC TRAINING
training_input = reshape(principle_spikes,[],1);


% store mean and standard deviation - same need to be used to normalise test data
modelParameters.mu = mu;
modelParameters.sig = sig;
 
fprintf("Extracted PCA parameters.\n"); toc;


tic;
% reorder the data such that the cells are sorted from longest duration to
% shortest duration. LSTM automatically pads 'chunks' of the cells, so
% sorting them by length reduces the amount of padding. The input and
% output cells are sorted into matching order 
sequenceLengths = zeros(numel(training_input), 1);
for i=1:numel(training_input)
    sequence = training_input{i};
    sequenceLengths(i) = size(sequence,2);
end
[~,idx] = sort(sequenceLengths,'descend');
training_velocities = training_velocities(idx);
training_input = training_input(idx);

% Setting up parameters for network 
numFeatures = size(training_input{1},1);
numResponses = size(training_velocities{1},1);

miniBatchSize = 8; % chosen to evenly divide the training data. The data will be padded to ensure that all data sets in each minibatch have the same length. 
modelParameters.miniBatchSize = miniBatchSize;
numHiddenUnits = 50;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 5;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Verbose',0);
modelParameters.nnet = trainNetwork(training_input,training_velocities,layers,options);
toc;
% %% DIRECTION SPECIFIC TRAINING
% 
% % store mean and standard deviation - same need to be used to normalise test data
% modelParameters.mu = mu;
% modelParameters.sig = sig;
%  
% fprintf("Extracted PCA parameters.\n"); toc;
% 
% 
% for k_it = 1:N_directions
% % reorder the data such that the cells are sorted from longest duration to
% % shortest duration. LSTM automatically pads 'chunks' of the cells, so
% % sorting them by length reduces the amount of padding. The input and
% % output cells are sorted into matching order
%     dir_input = cell(N_trials, 1);
%     dir_velocities = cell(N_trials, 1);
%     for n = 1:N_trials
%         dir_input{n} = principle_spikes{n, k_it};
%         dir_velocities{n} = training_velocities{((k_it-1)*N_trials) + n};
%     end
%     sequenceLengths = zeros(numel(dir_input), 1);
%     for i=1:numel(dir_input)
%         sequence = dir_input{i};
%         sequenceLengths(i) = size(sequence,2);
%     end
%     [~,idx] = sort(sequenceLengths,'descend');
%     dir_velocities = dir_velocities(idx);
%     dir_input = dir_input(idx);
%     
%     % Setting up parameters for network 
%     numFeatures = size(dir_input{1},1);
%     numResponses = size(dir_velocities{1},1);
%     
%     miniBatchSize = 8; % chosen to evenly divide the training data. The data will be padded to ensure that all data sets in each minibatch have the same length. 
%     modelParameters.miniBatchSize = miniBatchSize;
%     numHiddenUnits = 200;
%     layers = [ ...
%         sequenceInputLayer(numFeatures)
%         lstmLayer(numHiddenUnits,'OutputMode','sequence')
%         fullyConnectedLayer(50)
%         dropoutLayer(0.5)
%         fullyConnectedLayer(numResponses)
%         regressionLayer];
%     
%     maxEpochs = 60;
%     
%     options = trainingOptions('adam', ...
%         'MaxEpochs',maxEpochs, ...
%         'MiniBatchSize',miniBatchSize, ...
%         'InitialLearnRate',0.01, ...
%         'GradientThreshold',1, ...
%         'Shuffle','never', ...
%         'Verbose',0);
%     modelParameters.nnet{k_it} = trainNetwork(dir_input,dir_velocities,layers,options);
% end
end

function velocities = downsize_and_convert_to_velocity(x, W)
    
    n = size(x, 2) / W;
    velocities = zeros(2, n);
    for i = 1:n
        velocities(1,i) = x(1,i * W)-x(1,(W * (i - 1)) + 1);
        velocities(2,i) = x(2,i * W)-x(2,(W * (i - 1)) + 1); 
    end
    velocities = velocities/W;
    
end
  
% output: matrix, N_trials*N_directions x longest time sequence
function spike_rate_all_trials = make_av_spike_rate(training_input, N_trials, dir)
    % get all trials of chosen direction 
    i = 1;
    for n = (1+(N_trials*(dir-1))):(dir*N_trials)
        s{i} = training_input{n};
        i = i + 1;
    end
    % get length of longest time sequence
    max_duration = max(cellfun('size',training_input,2));
    % pad all sequences with 0s to have same length (max_duration)
    s = cellfun(@(x) padarray(x, [0, max_duration - size(x, 2)], 0, 'post'), s, 'un', 0);

    running_total = zeros(size(s{1}, 1), max_duration);
    for n = 1:size(s, 1)
        running_total = running_total + s{n};
    end
    spike_rate_all_trials = running_total/N_trials;
   
   
%     spike_rate_all_trials = spike_rate_all_trials(:,2:57);
end
 
function [Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, cutoff)


    % check if cutoff on eigenvalues within correct range
    if cutoff<0 || cutoff >1
        cutoff = 0.2;
    end
    
    A = cov(spike_rate_av_trials'); % 98x98 covariance matrix

    [V,D] = eig(A);
    [~,ind] = sort(diag(D), 'descend');
    Ds = D(ind, ind);
    Vs = V(:, ind);
    
    M = 5;
    
end



%%% LDA - Not Working or Complete Yet
% labels = repelem([1:1:N_directions]', size(spike_rate_av_trials,1), 1);
% all_spike_rate_av_trials = [all_spike_rate_av_trials labels];
% % remove all columns that only contain zeros 
% all_spike_rate_av_trials=all_spike_rate_av_trials(:,any(all_spike_rate_av_trials));
% 
% all_spike_rate_av_trials = all_spike_rate_av_trials(:,1:size(all_spike_rate_av_trials,2)-1);
% 
% % delete_id=[]; % vector of low variance columns
% % for i=1:size(all_spike_rate_av_trials,2)
% %   if(var(all_spike_rate_av_trials(:,i))==0) % if no variance in column 
% %          delete_id=[delete_id,i];
% %   end
% % end
% % all_spike_rate_av_trials(:,delete_id)=[];
% 
% % built-in LDA function
% Mdl = fitcdiscr(all_spike_rate_av_trials, labels, 'discrimType', 'linear');
%  
% A_l = Mdl.Sigma^(-1) * Mdl.BetweenSigma;
% modelParameters.CD_Mdl = Mdl;
%  
% %spikes_mean = mean(spike_rate, 2);
%  
% [V,D] = eig(A_l);
% [~,ind] = sort(diag(D), 'descend');
% Ds = D(ind, ind);
% Vs = V(:, ind);
%  
% cutoff=0.2;
% Is = find(diag(Ds)<max(Ds,[],'all')*cutoff);
% M = Is(1);
%
% V_red = Vs(:,1:M); % principal component vectors
% 
% modelParameters.M = M; % keep same M for all
% modelParameters.Vs = Vs;
% modelParameters.Ds = Ds;
% modelParameters.V_red = V_red;
%
% reshape to cell array of dimensions N_trials, N_directions
%
% for k_it =1:N_directions
%     i = 1;
%     for n = (1+(N_trials*(dir-1))):(dir*N_trials)
%         direction_specific_data{i} = training_input{n};
%         i = i + 1;
%     end
% principle spikes
% direc = cellfun(@(x) V_red'*x, training_input, 'UniformOutput', 0);
%
%
% mu = mean([training_input{:}],2); % mean spike rate for each neuronacross all trials for each directions
% sig = std([training_input{:}],0,2); % standard deviation of spike rate across all trials and directions
%
