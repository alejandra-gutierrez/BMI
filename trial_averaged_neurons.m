function [avg_neurons, time_axis] = trial_averaged_neurons(noTrials, binWidth)
load('monkeydata_training.mat'); 
% ---------------------------------------------------------------------- %   
    % Construct array that is the sum of the spike data of all of the 
    % trials at a certain direction for a certain neuron  
    time_axis = cell(1,8);
    avg_neurons = cell(98,8); 
%     max_x = 1; 
    for direction_no = [1:1:8]
        for neuron_no = [1:1:98]
            x = trial(1, direction_no).spikes(neuron_no,:); 
            for i = [2:1:noTrials] 
                y = trial(i, direction_no).spikes(neuron_no,:);
                if size(x, 2) > size(y, 2)
                    y(numel(x)) = 0; % fills 'missing' elements in y with 0 if y shorter than x
                elseif size(y, 2) > size(x, 2)
                    x(numel(y)) = 0; % fills 'missing' elements in x with 0 if x shorter than y
                end
                x = x + y; % x is the sum of spike data over all trials for specific direction and neuron 
                if mod(length(x), binWidth) ~= 0
                    x = [x zeros(1, binWidth - mod(length(x), binWidth))];
                end
                r = reshape(x, binWidth, []);
                if size(r, 1) > 1
                    r = sum(r);
                end
            end
            avg_neurons{neuron_no,direction_no} = (r/noTrials)*(1000/binWidth);
%             avg_neurons{neuron_no,direction_no} = (r/noTrials);

        end
        time_axis{1, direction_no} = [median([1:1:binWidth]):binWidth:length(x)];
    end
%     time_axis = [1:binWidth:max_x]/1000;
end
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

% ---------------------------------------------------------------------- %   