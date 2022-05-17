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
training_output = training_output(idx);
training_input = training_input(idx);

% alter sizing for binning the data. If the duration is not a multiple of
% windowsize, the extra data points are clipped off (these will be part of
% the final 100 data points and therefore redundant to position prediction.
% The same process is applied to both input and output training sets.
for n = 1:N
    clip = mod(size(training_input{n}, 2),windowsize);
    t = training_input{n};
    training_input{n} = [];
    training_input{n} = t(:,1:end-clip);
    t = training_output{n};
    training_output{n} = [];
    training_output{n} = t(:,1:end-clip);
end

% convert the output position data into velocity, and bin into chunks of
% size windowsize. Velocity of each bin calculated by subtracting first
% position value in bin from last in bin, and dividing by windowsize.
training_velocities = cellfun(@(x) downsize_and_convert_to_velocity(x, windowsize), training_output, 'un', 0);

% bin the spike data. spikes are summed in each bin, and converted to a
% rate in Hz. 
training_input = cellfun(@(x) (squeeze(sum(reshape(x,size(x,1),windowsize,[]),2))/windowsize)*1000, training_input, 'un', 0);



numFeatures = size(training_input{1},1);
numResponses = size(training_velocities{1},1);

mu = mean([training_input{:}],2); % mean spike rate across all trials and directions
sig = std([training_input{:}],0,2); % standard deviation of spike rate across all trials and directions

% store mean and standard deviation - same need to be used to normalise test data
modelParameters.mu = mu;
modelParameters.sig = sig;

% normalise training data
for i = 1:numel(training_input)
    training_input{i} = (training_input{i} - mu) ./ sig;
end

miniBatchSize = 10; % chosen to evenly divide the training data. The data will be padded to ensure that all data sets in each minibatch have the same length. 
modelParameters.miniBatchSize = miniBatchSize;
numHiddenUnits = 200;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
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
