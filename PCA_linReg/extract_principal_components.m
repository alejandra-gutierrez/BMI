function a = extract_principal_components(spikes, V_red)
    
spikes_mean = mean(spikes, 2);

    a = V_red'*(spikes - spikes_mean);

end