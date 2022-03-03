noTrials = 50; % number of trials to include - section for training data 
binWidth = 10; 
[all_nrns, time_axis] = trial_averaged_neurons(noTrials, binWidth); % create averages across all trials of each neuron in each direction 


% smoothingFactor: value of how many datapoints to group i.e. value of 10
% implies 10ms bin width 

% get data for a single neuron n in  direction k by all_nrns(((direction - 1)*98)+neuron,:)
% no smoothing/processing applied - just average of spike data 

axis = 1; % X direction movement 
all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth); 

% Neurons could be responsible for direction or acceleration?

%% PSTH 
figure
direction = 8;
neuron = 5;
binCenters = time_axis - 1/1000 + (binWidth/2000);
bar(binCenters, all_nrns(((direction - 1)*98)+neuron,:))
title("PSTH for Neuron " + neuron + " Direction k = " + direction)
ylabel("Spike Frequency (Hz)")
xlabel("Time (ms)")
%% X-direction tuning curves?
direction = 1;
indices = [];
all_psns = round(all_psns, 3); % round position to 3dp precision 
for direction = [1:1:8]
    for i = [1:1:length(all_psns(direction))]
        
    end
end

    
% loop through position vector 
% for each value of position, store index of that position 
% if mutliple occurences of same position, also save index 
    % e.g. position x occurs at indexes i1, i7, i9

% in spike data, sum elements of array of these indices
    % i.e. sum elements at i1, i7 and i9

for n = [1 5]
    figure 
    i = 1;
    for neuron = [1:1:10]
        subplot(5, 2, i)
        yFit = zeros(1,size(all_psns, 2));
        counter = 1;
        index = n;
        plot(all_psns(direction,:), all_nrns(((direction-1)*98)+neuron,:),'.')
        hold on 
        p = polyfit(all_psns(direction,:),all_nrns(((direction-1)*98)+neuron,:),n);
        while index >= 0
            yFit = yFit + all_psns(direction,:).^index*p(counter);
            index = index - 1;
            counter = counter + 1;
        end
        plot(all_psns(direction,:), yFit)
        xlabel("x position")
        ylabel("Spike Frequency (Hz)")
        title("Neuron: " + neuron)
        i = i + 1;
    end 
end
% if fit curve does not appear to be continuous line (i.e. loop can
% be seen), neuron does not code for position


%%
for direction = [1:1:8]
    figure 
    title("Relationship Between Neuron Spike Rate and X-Direction Position: k = " + direction)
    i = 1;
    for neuron = [1:1:98]
        subplot(14, 7, i)
        yyaxis left 
        plot(time_axis, all_nrns(((direction - 1)*98)+neuron,:))
        ylabel("Expected Number of Spikes in " + binWidth + "ms")
        yyaxis right 
        plot(time_axis, all_psns(direction,:))
        ylabel("Hand Position in X Direction")
        xlabel('Time (ms)') 
        title("Neuron: " + neuron)
        i = i + 1;
    end
end
%     