function spike_rates = get_spike_rates2(trials, windowsize, t_step)
    % output size spike_rates: cell [N_trials x N_angles]
%       each is a double, size [N_neurons  x (t_max_each/t_step)]

% input trials - trial(n, k), spikes size(N_neurons, t_max)
%input t_step: not there = assume keep all time steps
% if t_step == 0 -> assume want default size reduction = windowsize/2


    [N_trials, N_angles]= size(trials);
    N_neurons = size(trials(1,1).spikes,1);
    
    
    if ~exist('t_step', 'var') || isempty(t_step)
        t_step = 1;
    elseif t_step <=0
        t_step = floor(windowsize/2);
    end

    
    spike_rates = cell(N_trials, N_angles);
    %spike_rates = zeros(1500, N_trials, N_angles, N_neurons);
    
    for n = 1:N_trials
        for k = 1:N_angles
            spikes = trials(n,k).spikes;
            t_max = size(spikes, 2);
            spike_rates{n, k} = zeros(N_neurons, t_max);
            
            for neuron = 1:N_neurons
                for t = windowsize:t_step:t_max
%                     rate = spikes(neuron, t-windowsize+1:t)*ones(windowsize,1)/windowsize*1000;
                    rate = sum(spikes(neuron, t-windowsize+1:t))/windowsize*1000;
%                     rate = rate/windowsize*1000;
                    spike_rates{n, k}(neuron, floor(t/t_step) ) = rate;

                end
            end
        end
    end
    
end

