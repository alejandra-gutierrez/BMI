function out = singleNeuronManyTrials_Raster(neuron_no, noTrials, direction_no)
    load('monkeydata_training.mat');
    l = 1;
    figure
    for i = [1:1:noTrials]
        x = trial(i, direction_no).spikes(neuron_no,:);
        hold on 
        plot(i*x, '.', 'LineStyle', 'None')
        if size(x, 2) > l
            l = size(x,2);
        end
    end
    axis([0 l 1 size(trial, 1)])
    xlabel('Time (bins)')
    ylabel('Trial Number')
    title("Raster Plot for Neuron Number " + neuron_no + ", Direction " + direction_no)
    hold off 
end