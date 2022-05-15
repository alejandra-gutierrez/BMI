function [A, W, H, Q] = kalmanCoeffs(training_input, training_output)
    if iscell(training_input) || isstruct(training_input)
        error("Wrong input type: must be [N_components x time] matrix!");
    elseif size(training_input, 2) ~= length(training_output)
        error("The size of the training input and training output do not match:\n training_input: size %g-by%g  must be size %g-by-%g", size(training_input, 1), size(training_input, 2), size(training_output, 1), size(training_output, 2));
    end
%     time_lag = 75; % 75 ms used as multiple of windowsize and in predicted 'good' window (Wu 2003)
    % training input data is N_Neurons x T
    % training output is T x 3
    
    % assuming only two dimensions used
    M = size(training_input, 2);
    A = (training_output(:,2:end)*training_output(:,1:end-1)')/(training_output(:,1:end-1)*training_output(:,1:end-1)');

    W = 1/(M-1)*((training_output(:,2:end)*training_output(:,2:end)')-(A*(training_output(:,1:end-1)*training_output(:,2:end)')));

    H = (training_input*training_output')/(training_output*training_output');

    Q = 1/M*((training_input*training_input')-(H*(training_output*training_input')));    


end