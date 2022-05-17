function [x, y] = positionEstimator(test_data, modelParameters)
  

    N_trials_test = size(test_data, 1); % will be one, since test sequences passed in one at a time
    N_directions = size(test_data, 2); % will be one, since test sequences passed in one at a time
    N = N_trials_test * N_directions;
    windowsize = modelParameters.windowsize;
    
    % initialise cell arrays - only one data sequence, but keeps consistent
    % formatting for model (test data same as training data)
    test_input = cell(N,1);

    % add spike data to cell array 
    for n = 1:N
        current_row = mod(n-1, N_trials_test)+1;
        current_column = fix((n - 1)/N_trials_test) + 1;
        test_input{n} = test_data(current_row, current_column).spikes;
    end

    % sort spike data by length - not really necessary since only one
    % sequence
    numObservationsTest = numel(test_input);
    sequenceLengthsTest = zeros(numel(test_input), 1);
    for i=1:numObservationsTest
        sequence = test_input{i};
        sequenceLengthsTest(i) = size(sequence,2);
    end
    [~,idx] = sort(sequenceLengthsTest);
    test_input = test_input(idx);

    % clip last data points to ensure length is multiple of windowsize
    for n = 1:N
        clip = mod(size(test_input{n}, 2),windowsize);
        t = test_input{n};
        test_input{n} = [];
        test_input{n} = t(:,1:end-clip);
    end

    % bin input data, convert rate to Hz
    test_input = cellfun(@(x) (squeeze(sum(reshape(x,size(x,1),windowsize,[]),2))/windowsize)*1000, test_input, 'un', 0);

    % normalise spike data with same values as training data
    for i = 1:numel(test_input)
        test_input{i} = (test_input{i} - modelParameters.mu) ./ modelParameters.sig;
    end

    miniBatchSize = modelParameters.miniBatchSize;

    % run model to estimate velocity sequence
    for m=1:N_trials_test
        test_data(m).decodedHandPos = [];
        estimated_velocity = predict(modelParameters.nnet, test_input,'MiniBatchSize',miniBatchSize, 'SequenceLength','longest');

        x(m) = sum(estimated_velocity{1}(1,:)) + test_data(m).startHandPos(1);
        y(m) = sum(estimated_velocity{1}(2,:)) + test_data(m).startHandPos(2);

    end
end
