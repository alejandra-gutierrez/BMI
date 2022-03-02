function dir = position_estimation(spikes, neuron_tuning, unit_vect_list)
    
    neuron_resp = sum(spikes, 2)/size(spikes, 2);
    [Max_tune, I] = max(neuron_tuning, [], 2);
    
    fa = neuron_resp./ Max_tune; 
    tune_vect_list = unit_vect_list(:,I)'; 
    orientation_vect = ones(1, size(fa,1))*(fa.*tune_vect_list);
    dir = atan(orientation_vect(2)/orientation_vect(1));
    

end