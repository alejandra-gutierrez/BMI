function manyNeuronsSingleTrial_Raster(noNeurons, trial_no, direction_no)
    load('monkeydata_training.mat');
    figure
    for i = [1:1:noNeurons]
        x = trial(trial_no, direction_no).spikes(i,:);
        hold on 
        plot(i*x, 'b.', 'LineStyle', 'None')
    end
    axis([0 size(x, 2) 1 noNeurons])
    xlabel('Time (bins)')
    ylabel('Neuron Number')
    title("Raster Plot for Trial No. " + trial_no + ", Direction " + direction_no)
    hold off 
end