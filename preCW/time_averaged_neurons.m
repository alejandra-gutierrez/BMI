function avg_neurons = trial_averaged_neurons(noTrials)
s = 1; 

load('monkeydata_training.mat'); 
% ---------------------------------------------------------------------- %   
    % Construct array that is the sum of the spike data of all of the 
    % trials at a certain direction for a certain neuron  
    avg_neurons = []; % NOTE: tested and faster to assign empty than loop to preallocate to required size 
        
    for direction_no = [1:1:8]
        for neuron_no = [1:1:98]
            x = trial(1, direction_no).spikes(neuron_no,:); 
            for i = [1:1:noTrials] 
                y = trial(i, direction_no).spikes(neuron_no,:);
                if size(x, 2) > size(y, 2)
                    y(numel(x)) = 0; % fills 'missing' elements in y with 0 if y shorter than x
                elseif size(y, 2) > size(x, 2)
                    x(numel(y)) = 0; % fills 'missing' elements in x with 0 if x shorter than y
                end
                x = x + y;
            end
            avg_neurons(((direction_no-1)*98)+neuron_no,1:length(x)) = x;
            % first 98 rows contain neuron data corresponding to direction
            % 1, second 98 to direction 2, etc... i.e. 784 rows = 98
            % neurons * 8 directions
        end
    end
end
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

% ---------------------------------------------------------------------- %   