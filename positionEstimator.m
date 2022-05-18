function [x, y] = positionEstimator(test_data, modelParameters)
  

    N_trials_test = size(test_data, 1); % will be one, since test sequences passed in one at a time
    N_directions = size(test_data, 2); % will be one, since test sequences passed in one at a time
    N = N_trials_test * N_directions;
    windowsize = modelParameters.windowsize;
    t_pre_mvt = modelParameters.t_pre_mvt;
    t_mvt = modelParameters.t_mvt;
    mu = modelParameters.mu;
    sig = modelParameters.sig;

    % initialise cell arrays - only one data sequence, but keeps consistent
    % formatting for model (test data same as training data)
    test_input = cell(N,1);

    % add spike data to cell array 
    for n = 1:N
        current_row = mod(n-1, N_trials_test)+1;
        current_column = fix((n - 1)/N_trials_test) + 1;
        test_input{n} = test_data(current_row, current_column).spikes;
    end



    % clip last data points to ensure length is multiple of windowsize
    for n = 1:N
        clip = mod(size(test_input{n}, 2),windowsize);
        t = test_input{n};
        test_input{n} = [];
        test_input{n} = t(:,t_mvt+1:end-clip);
    end

    % bin input data, convert rate to Hz
    test_input = cellfun(@(x) (squeeze(sum(reshape(x,size(x,1),windowsize,[]),2))/windowsize)*1000, test_input, 'un', 0);



    miniBatchSize = modelParameters.miniBatchSize;

    % run model to estimate velocity sequence
    for m=1:N_trials_test
        test_data(m).decodedHandPos = [];
        t_end = size(test_input, 2);

        % STEP 1: COMPUTE PREDICTED DIRECTION
        sr = sum(test_data(m).spikes(:, 1:t_pre_mvt), 2)'; % [1 x N_neurons]
        dir = predict(modelParameters.knn, sr); % using toolbox

        % STEP 2: COMPUTE CURRENT POSITION
        V_red = modelParameters.V_red{dir};
        M = modelParameters.M{dir};

        % only one test trial passed in at a time - hardcoding should be
        % fine
        principle_spikes = V_red'*test_input{1};
    
        % normalise spike data with same values as training data
        principle_spikes = (principle_spikes - mu(:, dir)) ./ sig(:,dir);
%         test_input = reshape(principle_spikes,[],1);

         % sort spike data by length - not really necessary since only one
        % sequence

        estimated_velocity = predict(modelParameters.nnet, principle_spikes,'MiniBatchSize',miniBatchSize, 'SequenceLength','longest');

        x(m) = sum(estimated_velocity(1,:)) + test_data(m).startHandPos(1);
        y(m) = sum(estimated_velocity(2,:)) + test_data(m).startHandPos(2);

    end
end
