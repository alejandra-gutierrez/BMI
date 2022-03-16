function weights = linearRegression(direction, training_inputs, training_outputs)
% output will give vector of the weights of a linear regression algorithm
% to predict (position/velocity/acceleration) (at a point in time?) using
% inputs and output data from 98 neurons 

% format of data: direction = k [1:8]
% training_input - spike_data
%   format as cell array
%   [98 neurons x 8 directions]
%   each cell array as average spike rates

% training_inputs should be a matrix of 98 columns and T rows where T is
% the length (number of bins) of spike data. The elements will be the spike
% rate for that neuron in that time bin. 

% training_outputs should be a vector of length T containing the
% (position/velocity/acceleration) for chosen axis (x or y data) in each
% time bin.

    weights = [];
    d = transpose(training_outputs{direction});
    neuron = 1;
    U = zeros(length(training_inputs{neuron, direction}),98);
    for neuron = 1:1:98
        U(:,neuron) = training_inputs{neuron, direction};
    end

    
%     A = transpose(U)*U;
%     B = inv(A);
%     C = A*B;
%     W = C*d;
%     Ut = transpose(U);
%     B = U*Ut;
%     C = inv(B);
%     D = C*Ut*d;
%     weights = (Ut*U)\Ut*d;
    weights = regress(d, U);
    

end




