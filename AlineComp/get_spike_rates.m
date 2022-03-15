function spike_rates = get_spike_rates(trials, windowsize)
    
    [N_trials, N_angles]= size(trials);
    N_neurons = size(trials(1,1).spikes,1);
    
    % size spike_rates: [max_t x N_trials x N_angles x N_neurons]
    
    
    spike_rates = zeros(1500, N_trials, N_angles, N_neurons);
    
    max_t = 0;


    for n = 1:N_trials
        for k = 1:N_angles
            spikes = trials(n,k).spikes;
            timesteps = size(spikes, 2);
            if timesteps>max_t
                max_t = timesteps;
            end
            
            for t = floor(1+windowsize/2:timesteps-windowsize/2)
                for neuron = 1:N_neurons
                    local_spikes = spikes(neuron, t-ceil(windowsize/2):t+floor(windowsize/2));
                    spike_rates(t, n, k, neuron) = sum(local_spikes)/windowsize*1000;
                end
            end
        end
    end

    spike_rates(max_t:end,:,:) = [];
    
end