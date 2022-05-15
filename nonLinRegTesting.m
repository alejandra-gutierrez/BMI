
neuron = 1;
direction = 1;

X = zeros(length(all_nrns{neuron, direction}),98); % predictors - 98 neurons, T time 'chunks' - T x 98 array 
for neuron = 1:1:98
    X(:,neuron) = all_nrns{neuron, direction};
end
Y = transpose(all_psns{direction}); % responses i.e. output data from training trials

% modelfun = ; % hougen watson model 
beta0 = [1 1 1 1 1];
weights = nlinfit(X, Y, @hougen, beta0);

ypred = hougen(weights, 