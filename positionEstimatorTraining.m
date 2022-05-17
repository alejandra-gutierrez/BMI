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

for k_it = 1:N_directions
    spike_rate_av_trials = make_av_spike_rate(training_input, N_trials, k_it);
    [Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, proportion);
    dir = k_it;
    
    V_red = Vs(:, 1:M);
    modelParameters.M{k_it} = M; % keep same M for all
    modelParameters.dir{k_it} = dir;
    modelParameters.Vs{k_it} = Vs;
    modelParameters.Ds{k_it} = Ds;
    modelParameters.V_red{k_it} = V_red;
 
    mu(:,k_it) = mean([direction_specific_data{:,k_it}],2); % mean spike rate for each neuron across all trials for each direction
    sig(:, k_it) = std([direction_specific_data{:,k_it}],0,2); % standard deviation of spike rate for each neuron across all trials for each direction
    % normalise
    for n = 1:N_trials
        direction_specific_data{n,k_it} = (direction_specific_data{n, k_it}-mu(:,k_it))./sig(:, k_it);
    end

    for n = 1:N_trials
        principle_spikes{n, k_it} = V_red'*direction_specific_data{n, k_it};
    end
end
training_input = reshape(principle_spikes,[],1);

% store mean and standard deviation - same need to be used to normalise test data
modelParameters.mu = mu;
modelParameters.sig = sig;
 
fprintf("Extracted PCA parameters.\n"); toc;

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


%% OLD %% 
% t_start = tic;
% tic;
% [N_trials_tr, N_angles] = size(training_data);
% N_neurons = size(training_data(1).spikes, 1);
%  
% windowsize = 20;
% t_mvt = 210;
% t_pre_mvt = 300;
% t_step = windowsize/2;
% t_step = ceil(t_step);
% n_neighbours = 12;
% proportion = 2/100; % th for selection of principal components
%  
% fprintf("\nFinding spike rates and velocities...");
% [velx_tr, vely_tr, ~] = getvel2(training_data, windowsize, t_step, t_mvt);
% spike_rate = get_spike_rates2(training_data, windowsize, t_step, t_mvt);
% fprintf("Spike_Rate done...\n");
% toc;
%  
% %% TRAIN KNN MODEL
%  
% fprintf("Training KNN model...");
% spikesr = zeros(N_angles*N_trials_tr, N_neurons);
% labels = zeros(1, N_angles*N_trials_tr);
% for k_it = 1:N_angles
%     for n_it = 1:N_trials_tr
%          spikesr( (k_it-1)*N_trials_tr + n_it, :) = sum(training_data(n_it, k_it).spikes(:, 1:t_pre_mvt), 2)';           
%         labels( (k_it-1)*N_trials_tr + n_it) = k_it;
%     end
% end
%  
% % knn = fitcknn(spikesr, labels);
% for k_it = 1:N_angles+1
%     modelParameters(k_it).KNNSpikesr = spikesr;
%     modelParameters(k_it).KNNLabels = labels;
%     % modelParameters(k_it).knn = knn;
%     modelParameters(k_it).n_neighbours = n_neighbours;
% end
% fprintf("KNN model done. "); toc;
%  
%  
% %% TRAIN POSITION ESTIMATOR
% fprintf("Extracting Principal component vectors from data...");
%  
% all_spike_rate_av_trials = [];
% for k_it =0:N_angles
%     spike_rate_av_trials = make_av_spike_rate(spike_rate, k_it);
%     size(all_spike_rate_av_trials);
%     size(spike_rate_av_trials);
%     all_spike_rate_av_trials = [all_spike_rate_av_trials; spike_rate_av_trials];
% end
%  
% labels = repelem([0:1:8]', size(spike_rate_av_trials,1), 1);
% all_spike_rate_av_trials = [all_spike_rate_av_trials labels];
%  
% %size(all_spike_rate_av_trials) %882x80
% %size(labels) %882x1
%  
% % remove zero within-class variances
% %all_spike_rate_av_trials = all_spike_rate_av_trials(:,any(all_spike_rate_av_trials))
% size(all_spike_rate_av_trials);
%  
% % built-in LDA function
% Mdl = fitcdiscr(all_spike_rate_av_trials(:,1:size(all_spike_rate_av_trials,2)-1), labels, 'discrimType', 'Linear');
%  
% A_l = Mdl.Sigma^(-1) * Mdl.BetweenSigma;
% modelParameters(1).Mdl = Mdl;
%  
% %spikes_mean = mean(spike_rate, 2);
%  
% [V,D] = eig(A_l);
% [d,ind] = sort(diag(D), 'descend');
% Ds = D(ind, ind);
% Vs = V(:, ind);
%  
% cutoff=0.2;
% Is = find(diag(Ds)<max(Ds,[],'all')*cutoff);
% M = Is(1);
% V_red = Vs(:,1:M); % principal component vectors
%  
%  
% for k_it =0:N_angles
%     spike_rate_av_trials = make_av_spike_rate(spike_rate, k_it);
%     [~, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, proportion);
%     dir = k_it;
%     if k_it == 0
%         k_it = N_angles+1;
%     end
%     V_red = Vs(:, 1:M);
%     modelParameters(k_it).M = M; % keep same M for all
%     modelParameters(k_it).dir = dir;
%     modelParameters(k_it).Vs = Vs;
%     modelParameters(k_it).Ds = Ds;
%     modelParameters(k_it).V_red = V_red;
%  
%     for n_it = 1:N_trials_tr
%         if k_it == N_angles+1
%             for k = 1:N_angles
%                 % make a specific array for non-specific training
%                 spikes_mean = mean(spike_rate{n_it, k}, 2);
%                 principal_spikes_0{n_it, k} = V_red'*(spike_rate{n_it, k}-spikes_mean);
%             end
%         else
%            spikes_mean = mean(spike_rate{n_it, k_it}, 2);
%            principal_spikes_tr{n_it, k_it} = V_red'*(spike_rate{n_it, k_it}-spikes_mean);
%         end
%         fprintf(".");
%     end
%     fprintf("\n"); 
% end
%  
% fprintf("Extracted LDA parameters.\n"); toc;
%  
%  
% fprintf("Starting Neural Networks Training.\t");
% % x_results_dir = cell(1, 9);
% % y_results_dir = cell(1, 9);
%  
% for k_it = 1:N_angles
%     fprintf("k=%g.\t", k_it);
%     if (k_it ==0) % non-direction specific training
%         [input_datax, output_datax] = linearizeInputOutput(principal_spikes_0, velx_tr, k_it);
%         [input_datay, output_datay] = linearizeInputOutput(principal_spikes_0, vely_tr, k_it);
%  
% % % % ---------- CODE TO FIND MODEL AND PARAMETERS ------------ % 
% %         [~] = fitrauto(input_datax', output_datax,"Learners", "net","HyperparameterOptimizationOptions",struct("Optimizer","asha","MaxTime", 2000,"Repartition", true,"MinTrainingSetSize", 30000));
% % %         disp("x results, k = 0 done")
% %         [~] = fitrauto(input_datay', output_datay,"Learners", "net","HyperparameterOptimizationOptions",struct("Optimizer","asha","MaxTime", 2000,"Repartition", true,"MinTrainingSetSize", 30000));
% % %         disp("y results, k = 0 done")
% % % 
% % %        
% % %         x_results_dir{1, 9} = x_results;
% % %         y_results_dir{1, 9} = y_results;
% % % % --------------------------------------------------------- % 
% %  
% % % ------------------ 11.05.2022 Test 1 -------------------- % 
%         x_activation = "relu";
%         y_activation = 'relu';
%         x_lambda = 0.5e-6;
%         y_lambda = 0.5e-6;
%         x_standardize = false;
%         y_standardize = false;
%         x_layers = [10 5];
%         y_layers = [10 5];
% 
%       
% % % --------------------------------------------------------- % 
%  
% % % ------------------ 11.05.2022 Test 2 -------------------- % 
% %         x_activation = 'none';
% %         y_activation = 'none';
% %         x_lambda = 2.26216301188866e-10;
% %         y_lambda = 1.84010252449154e-10;
% %         x_standardize = false;
% %         y_standardize = false;
% %         x_layers = [1 4 170];
% %         y_layers = [1 33];
% % % --------------------------------------------------------- % 
%  
%         mdl_x = fitrnet(input_datax, output_datax,'Activations', x_activation, 'Standardize', x_standardize,'OptimizeHyperparameters',['Lambda', 'LayerSizes'], 'ObservationsIn','columns');
%         mdl_y = fitrnet(input_datay, output_datay,'Activations', y_activation, 'Standardize', y_standardize, 'OptimizeHyperparameters',['Lambda', 'LayerSizes'],'ObservationsIn','columns');
%         
%  
%         k_it = N_angles+1;
%     else  % direction specific training
%         [input_datax, output_datax] = linearizeInputOutput(principal_spikes_tr, velx_tr, k_it);
%         [input_datay, output_datay] = linearizeInputOutput(principal_spikes_tr, vely_tr, k_it);
%  
%         x_activation = ["tanh", "none"];
%         y_activation = 'tanh';
%         x_lambda = 0.5e-4;
%         y_lambda = 0.5e-7;
%         x_standardize = false;
%         y_standardize = false;
%         x_layers = [3 10];
%         y_layers = [10];
% % % ---------- CODE TO FIND MODEL AND PARAMETERS ------------ % 
% %         [~, x_results] = fitrauto(input_datax', output_datax,"Learners", ["gp", "net", "linear"],"HyperparameterOptimizationOptions",struct("Optimizer","asha","MaxTime", 600));
% %         disp("x results, k = "+k_it+" done")
% %         [~, y_results] = fitrauto(input_datay', output_datay, "Learners", ["gp", "net", "linear"],"HyperparameterOptimizationOptions",struct("Optimizer","asha","MaxTime", 600));
% %         disp("y results, k = "+k_it+" done")
% % 
% %         x_results_dir{1, k_it} = x_results;
% %         y_results_dir{1, k_it} = y_results;
% % % --------------------------------------------------------- % 
%  
%  
%         mdl_x = fitrnet(input_datax, output_datax,'Activations', x_activation, 'Lambda', x_lambda, 'Standardize', x_standardize, 'LayerSizes', x_layers, 'ObservationsIn','columns');
%         mdl_y = fitrnet(input_datay, output_datay,'Activations', y_activation, 'Lambda', y_lambda, 'Standardize', y_standardize, 'LayerSizes', y_layers, 'ObservationsIn','columns');
%  
% %         mdl_x = fitrnet(input_datax, output_datax,'Activations', x_activation, 'Standardize', x_standardize,'OptimizeHyperparameters',{'Lambda', 'LayerSizes'}, 'ObservationsIn','columns');
% %         mdl_y = fitrnet(input_datay, output_datay,'Activations', y_activation, 'Standardize', y_standardize, 'OptimizeHyperparameters',{'Lambda', 'LayerSizes'},'ObservationsIn','columns');
% 
%  
%     end
%  
%     modelParameters(k_it).x = mdl_x;
%     modelParameters(k_it).y = mdl_y;
%  
% end
% fprintf("\n Done.\n");
% fprintf("Model Parameters:\n");
% % print model parameters
% for k_it = 1:N_angles+1
% %     M = modelParameters(k_it).M;
% %     dir = modelParameters(k_it).dir;
%     V_red = modelParameters(k_it).V_red;
% %     Vs = modelParameters(k_it).Vs;
% %     Ds = modelParameters(k_it).Ds;
% %     fprintf("dir=%g, M=%g,  size V_red=[%g, %g], size wX=[%g,%g], size wY=[%g,%g]\n",...
% %     dir, M, size(V_red,1),size(V_red,2), size(wX,1), size(wX, 2), size(wY,1), size(wY, 2));
% end
% fprintf("\nFinished Training.\n");
% toc; fprintf("\n");
% end
%  
%  
% %% ------------------ 11.05.2022 Test 1 -------------------- % 
% % 
% %         % 20220511_test_1: overlapping windows, removal of factor of 2 in getvel2.m
% %         switch k_it
% %             case 1 
% %                 x_activation = 'sigmoid';
% %                 y_activation = 'sigmoid';
% %                 x_lambda = 5.26126101367222e-09;
% %                 y_lambda = 2.39900825370355e-10;
% %                 x_standardize = false;
% %                 y_standardize = false;
% %                 x_layers = [158];
% %                 y_layers = [122 10];
% %             case 2
% %                 x_activation = 'relu';
% %                 y_activation = 'sigmoid';
% %                 x_lambda = 7.38069993914245e-06;
% %                 y_lambda = 0.000250739701973964;
% %                 x_standardize = true;
% %                 y_standardize = false;
% %                 x_layers = [1];
% %                 y_layers = [7 2];
% %             case 3
% %                 x_activation = 'tanh';
% %                 y_activation = 'tanh';
% %                 x_lambda = 0.01962600028231;
% %                 y_lambda = 0.00594350936327051;
% %                 x_standardize = false;
% %                 y_standardize = false;
% %                 x_layers = [8];
% %                 y_layers = [4 2 1];
% %             case 4 
% %                 x_activation = 'relu';
% %                 y_activation = 'tanh';
% %                 x_lambda = 0.0263093692996972;
% %                 y_lambda = 3.67759968227727e-09;
% %                 x_standardize = true;
% %                 y_standardize = false;
% %                 x_layers = [34];
% %                 y_layers = [1 1];
% %             case 5
% %                 x_activation = 'sigmoid';
% %                 y_activation = 'none';
% %                 x_lambda = 0.000268657518635669;
% %                 y_lambda = 4.67539961000468e-10;
% %                 x_standardize = true;
% %                 y_standardize = true;
% %                 x_layers = [4];
% %                 y_layers = [31];          
% %             case 6
% %                 x_activation = 'sigmoid';
% %                 y_activation = 'tanh';
% %                 x_lambda = 8.20101480235253e-08;
% %                 y_lambda = 0.0000011217772702795;
% %                 x_standardize = false;
% %                 y_standardize = true;
% %                 x_layers = [15 6];
% %                 y_layers = [2];         
% %             case 7
% %                 x_activation = 'sigmoid';
% %                 y_activation = 'tanh';
% %                 x_lambda = 7.98627128202956e-07;
% %                 y_lambda = 7.79798493638216e-07;
% %                 x_standardize = true;
% %                 y_standardize = true;
% %                 x_layers = [1 2];
% %                 y_layers = [1];            
% %             case 8
% %                 x_activation = 'tanh';
% %                 y_activation = 'none';
% %                 x_lambda = 5.20348980151386e-06;
% %                 y_lambda = 7.36951785331708e-07;
% %                 x_standardize = false;
% %                 y_standardize = true;
% %                 x_layers = [3];
% %                 y_layers = [14 37];
% %         end
% % % --------------------------------------------------------- % 
%  
% % % ------------------ 11.05.2022 Test 2 -------------------- % 
% % 
% %         % 20220511_test_2: non overlapping windows, removal of factor of 2 in getvel2.m
% %         switch k_it
% %             case 1 
% %                 x_activation = 'tanh';
% %                 y_activation = 'none';
% %                 x_lambda = 0.0000942272116847018;
% %                 y_lambda = 0.00128472775650572;
% %                 x_standardize = false;
% %                 y_standardize = false;
% %                 x_layers = [1];
% %                 y_layers = [1];
% %             case 2
% %                 x_activation = 'none';
% %                 y_activation = 'relu';
% %                 x_lambda = 1.43337640569499E-10;
% %                 y_lambda = 0.00346132340253797;
% %                 x_standardize = true;
% %                 y_standardize = false;
% %                 x_layers = [14];
% %                 y_layers = [4 48];
% %             case 3
% %                 x_activation = 'none';
% %                 y_activation = 'none';
% %                 x_lambda = 5.08428822263802E-06;
% %                 y_lambda = 0.00538886165573154;
% %                 x_standardize = false;
% %                 y_standardize = false;
% %                 x_layers = [5];
% %                 y_layers = [1];
% %             case 4 
% %                 x_activation = 'tanh';
% %                 y_activation = 'sigmoid';
% %                 x_lambda = 0.000083031624345349;
% %                 y_lambda = 0.0000628000050651263;
% %                 x_standardize = false;
% %                 y_standardize = true;
% %                 x_layers = [7 1 2];
% %                 y_layers = [1];
% %             case 5
% %                 x_activation = 'tanh';
% %                 y_activation = 'sigmoid';
% %                 x_lambda = 0.000351472700964922;
% %                 y_lambda = 8.30640481876404E-06;
% %                 x_standardize = false;
% %                 y_standardize = true;
% %                 x_layers = [4 1];
% %                 y_layers = [1];          
% %             case 6
% %                 x_activation = 'tanh';
% %                 y_activation = 'tanh';
% %                 x_lambda = 0.0151288459968471;
% %                 y_lambda = 0.000140169854149635;
% %                 x_standardize = false;
% %                 y_standardize = false;
% %                 x_layers = [8];
% %                 y_layers = [2];         
% %             case 7
% %                 x_activation = 'tanh';
% %                 y_activation = 'sigmoid';
% %                 x_lambda = 0.000216090599991575;
% %                 y_lambda =0.000820791821519997;
% %                 x_standardize = false;
% %                 y_standardize = false;
% %                 x_layers = [79 1 2];
% %                 y_layers = [6];            
% %             case 8
% %                 x_activation = 'relu';
% %                 y_activation = 'none';
% %                 x_lambda = 0.0311012435072188;
% %                 y_lambda = 0.00129477280992996;
% %                 x_standardize = false;
% %                 y_standardize = true;
% %                 x_layers = [10 172];
% %                 y_layers = [2 3];
% %         end
% % % --------------------------------------------------------- % 
% 
