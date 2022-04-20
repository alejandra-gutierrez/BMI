function neuron_weights = nonLinReg(training_input, training_output)
    disp(size(training_input));
    disp(size(training_output));

if iscell(training_input) || isstruct(training_input)
    error("Wrong input type: must be [N_components x time] matrix!");
elseif size(training_input, 2) ~= length(training_output)
    error("The size of the training input and training output do not match:\n training_input: size %g-by%g  must be size %g-by-%g", size(training_input, 1), size(training_input, 2), size(training_output, 1), size(training_output, 2));
end


    % Let's get to business!
    neuron_weights = []; % initialize weights that will be assigned to each input variable
    % weights linearly relate input variables to output variables
    
    % U is the input matrix
    % rows are input vectors u from the training set
    
    beta0 = zeros(5* size(training_input, 1), 1);

    disp(size(beta0));
    
    neuron_weights = nlinfit(training_input, training_output, @nonLinModelFun, beta0);

end
