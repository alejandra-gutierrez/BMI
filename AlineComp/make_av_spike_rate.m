function spike_rate_all_trials = make_av_spike_rate(spike_rates, dir)

% input format:
% neuronal spike rate data
    %   cell array {N_trials, k_dir} x [N_neurons x t_max_each]
% output:
%   spike_rate_all_trials, format [N_neurons  x t_max_all]
    
    [N_trials, N_angles] = size(spike_rates);
    N_neurons = size(spike_rates{1}, 1);
    
    if ~exist('dir', 'var') || isempty(dir) || dir == 0
        spike_rates = spike_rates(:); % no determined direction
        dir = 1;
    end
    if (size(spike_rates, 2) == 1)
        dir = 1; % deal with the function if direction already filtered out so no bug when accessing dir
    end
    
    % initialize memory allocation
    size_alloc = 2000;
    spike_rate_all_trials = zeros(N_neurons, size_alloc);
    
    max_t = 0;
    for n = 1:N_trials
        s = spike_rates{n, dir};
        timesteps = size(s, 2);
        if timesteps > max_t
            max_t = timesteps; % maximal time step
        end
        s(N_neurons, size_alloc) = 0; % zero pad
        
        spike_rate_all_trials  = spike_rate_all_trials + s;
    end
    spike_rate_all_trials(:, max_t+1:end)=[]; % reduce extra unused space
    
    spike_rate_all_trials = spike_rate_all_trials/N_trials;
end