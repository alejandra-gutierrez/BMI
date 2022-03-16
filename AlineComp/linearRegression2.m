function neuron_weights = linearRegression2(axis, training_input, training_output)
    % axis: 1, 2, 3 for x, y, z respectively
    
    % does not assume prior knowledge on direction of motion
    % (if did assume previous knowledge we would train model with only
    % specific k_dir)
    
    % training input: neuronal spike data
    %   {N_trials, k_dir} x [N_neurons x t_max]
    % training output: hand movement -- trial.handPos
    %   {N_trials, k_dir} x [3 x t_max]

    

end