function neuron_weights = linearRegression2(training_input, training_output, dir)
    % axis: 1, 2, 3 for x, y, z respectively
    
    % does not assume prior knowledge on direction of motion
    % (if did assume previous knowledge we would train model with only
    % specific k_dir)
    
    % training input: neuronal spike rate data
    %   {N_trials, k_dir} x [N_neurons x t_max]
    % N_neurons might be a different value if using dimension reduction
    % training output: hand velocity in specific axis
    %   [N_trials, N_angle, t_max]
    
    % We can also input an averaged spike rate
    [N_trials, N_angles] = size(training_input);
    N_neurons = size(training_input{1}, 1);
    if ~exist('dir', 'var') || isempty(dir)
        % do we want to use prior assumption on direction of movemnet?
        % % if dir undefined: make cell vector linear(instead of 2D for each dir)!
        % and then make mean!!
        % otherwise select only dir valid (and will be linear vect)
        if size(training_input, 2)>1
            training_input = training_input(:);
        end
        if size(training_output, 2) >1
            [M,N,P] = size(training_output);
            training_output = reshape(training_output, [N*M, P]);
            % new size: vel_axis = [N_trial*N-angles x max_t] 
        end        
    else
        if size(training_input, 2)>1 % select correct dir only
            training_input = training_input(:, dir);
            % new size: [N_trials x 1] cell, [N_neurons x t_max_each]
        end
        if size(training_output, 2) > 1 % select correct dir only and squeeze matrix shape
            training_output = squeeze(training_output(:, dir, :));
            % new size: [N_trials x max_t]
        end
    end

    if (size(training_input, 1) ~= size(training_output, 1))
        error("wrong size!\nSize output:%g\nSize input: %g", size(training_output,1), size(training_input,2));
    end

    t_max_all = size(training_output, 2);

    % linearize arrays in time and trials
    training_input2 = zeros(N_neurons, N_trials*t_max_all);

    size(training_output)
    size(training_input)

    for n=1:size(training_input,1)
        spikes_trial = training_input{n};
        spikes_trial(N_neurons, t_max_all) =0; % zeros padding
        %size(spikes_trial)
        training_input2(:, 1+(n-1)*t_max_all : n*t_max_all) = spikes_trial;
    end
    training_output = reshape(training_output', [size(training_output,1)*t_max_all, 1]);
    % flatten array back to back

    if (length(training_output) ~= size(training_input2, 2))
        
        error("wrong size!\nSize output:%g\nSize input: %g", size(training_output,1), size(training_input2,2));
    else
        disp("All good so far....");
    end
    
    whos training_input2
    whos training_output

%     figure;
%     plot(training_output);
%     figure;
%     plot(training_input2');

    % Let's get to business!
    neuron_weights = []; % initialize weights that will be assigned to each input variable
    % weights linearly relate input variables to output variables
    
    % U is the input matrix
    % rows are input vectors u from the training set
    training_input = training_input2';
    neuron_weights = (training_input'*training_input)^(-1)*training_input'*training_output;


end