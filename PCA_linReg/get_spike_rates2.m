function spike_rates = get_spike_rates2(trials, windowsize, t_step, t_start)
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
        t_step = ceil(windowsize/2);
    end
    if ~exist('t_start', 'var') || isempty(t_start)
        t_start = 1;
    end
    
    t_start = t_start - 1; % offset for iteration
    
    % make sure these are integers!
    t_start = floor(max([t_start, 0]));
    windowsize = ceil(windowsize) ;
    t_step = ceil(max([t_step, 1]));

    spike_rates = cell(N_trials, N_angles);
    
    max_t = 0;

    for n = 1:N_trials
        for k = 1:N_angles
            spikes = trials(n,k).spikes;
            timesteps = size(spikes, 2);
            spike_rates{n, k} = zeros(N_neurons, ceil(timesteps/t_step));
            
            if timesteps>max_t
                max_t = timesteps;
            end
            
            for t = t_start+windowsize:t_step:timesteps
                rate = sum(spikes(:, t-windowsize+1:t), 2)/windowsize*1000;
                spike_rates{n, k}(:, ceil((t-t_start)/t_step)) = rate;

            end
        end
    end
    
end

