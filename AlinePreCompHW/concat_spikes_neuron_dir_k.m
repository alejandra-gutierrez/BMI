function spikes_neuron_dir = concat_spikes_neuron_dir_k(trial, neuron, dir)
    % Concatenates spikes for a single neural unit from datafile trial
    % Given a movement direction k
    % neural unit neuron

    % output: 2D matrix A(n,t)
    % n is trial number
    % t is time (in ms)

    N_trials = size(trial, 1);
    N_reaching_angles = size(trial, 2);

    N_neuralunits = size(trial(1,1).spikes, 1);

    n_list = 1:N_trials;
    
    

    spikes_neuron_dir =[];

    for n = n_list
        spikes = trial(n, dir).spikes(neuron, :);
    
    
        if size(spikes, 2) ~= size(spikes_neuron_dir, 2)
            if size(spikes, 2) < size(spikes_neuron_dir, 2)
                spikes = cat(2, spikes, zeros(1, size(spikes_neuron_dir,2)-size(spikes, 2)));
            elseif size(spikes, 2) > size(spikes_neuron_dir, 2)
                spikes_neuron_dir = cat(2, spikes_neuron_dir, zeros(size(spikes_neuron_dir,1 ), size(spikes, 2) - size(spikes_neuron_dir, 2)));
            end
        end
        spikes_neuron_dir = cat(1, spikes_neuron_dir, spikes);
    end

     
    
end