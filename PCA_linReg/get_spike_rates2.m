function spike_rates = get_spike_rates2(trials, windowsize)
    
    [N_trials, N_angles]= size(trials);
    N_neurons = size(trials(1,1).spikes,1);
    
    % size spike_rates: [max_t x N_trials x N_angles x N_neurons]
    
    spike_rates = cell(N_trials, N_angles);
    %spike_rates = zeros(1500, N_trials, N_angles, N_neurons);
    
    max_t = 0;


    for n = 1:N_trials
        for k = 1:N_angles
            spikes = trials(n,k).spikes;
            timesteps = size(spikes, 2);
            if timesteps>max_t
                max_t = timesteps;
            end
            spike_rates{n, k} = zeros(N_neurons, timesteps);
            
            for neuron = 1:N_neurons
                for t = 1+windowsize:1:timesteps
                    rate = sum(spikes(neuron, t-windowsize+1:t));
                    rate = rate/windowsize*1000;
                    spike_rates{n, k}(neuron, t) = rate;
                end
            end
        end
    end
    
end

