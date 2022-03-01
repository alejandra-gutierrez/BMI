function [neuron_tuning_dir_k_mean, neuron_tuning_dir_k_std] = neuron_tuning_dir_k(trial, neuron, dir, N_bins)
    % gives the tuning of the neuron for movement direction k
    % tuning is described as firing rate averaged over time
    % estimate of std 
    
    % time_vect variable allows to choose over what time period we evaluate
    % the spikes


    % matrix: gives [neuron tuning, std tune]

    


    N_trials = size(trial, 1);
    N_reaching_angles = size(trial, 2);

    N_neuralunits = size(trial(1,1).spikes, 1);
    
    neuron_spikes = concat_spikes_neuron_dir_k(trial, neuron, dir);

    

    neuron_spikes = neuron_spikes(:, 1:350);
    T = size(neuron_spikes, 2);
    Delt = T / N_bins;

    spikes_neuron_mean = mean(neuron_spikes, 1);
    spikes_neuron_var = var(neuron_spikes, 1);
    
    

    % make bins over time:

    for i = 1: N_bins
        sub_zone = spikes_neuron_mean(round(Delt*(i-1)+1) : round(Delt*i) );
        spikes_hist(i) = sum(sub_zone)/Delt*1000; % gives rate in Hz vs time t
    end

    
    f_smooth = smoothdata(spikes_hist, 'gaussian', N_bins/2);

    neuron_tuning_dir_k_mean = mean(neuron_spikes(:));
    neuron_tuning_dir_k_std = std(mean(neuron_spikes),1);
    neuron_tuning_dir_k_mean = mean(f_smooth);
%     neuron_tuning_dir_k_std = std(f_smooth);
    
    
%     plot(f_smooth, 'DisplayName', "dir="+dir);


end
