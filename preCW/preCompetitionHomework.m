load('monkeydata_training.mat')
% monkeydata_training.mat contains 100 x 8 fields 
% 100 trials, 8 directions
% each entry is struct that contains trialID, spikes, handPos
% spikes contains neural data - time discretised sequence of 0/1 indicating
% absence or presence of a spike in 1ms window time/bin 
% spike train of duration Tms is represented by 1xT vector
% spike trains recorded from 98 neural units 
% handPos contains hand position trace 

%% Create a population raster plot for a single trial 
% for trial 98:
trial_no = 4;
direction_no = 3;
noNeurons = 98; % 98 neurons is all neurons
manyNeuronsSingleTrial_Raster(noNeurons, trial_no, direction_no);

%% Compute and display a Raster plot for one neural unit over many trials 
neuron_no = 49; 
noTrials = 100; % max no. trials is 100
singleNeuronManyTrials_Raster(neuron_no, noTrials, 1);

%% Compute peri-stimulus time histograms for different neural units 

% A peri-stimulus time histogram is a way to compare two sequences of
% events (usually action potentials) and to determine whether there is a
% relationship between the timing of events in one sequence and the the
% timing of events in the other sequence. For example, you might have
% intracellular recordings from two motoneurons and want to determine
% whether the two motoneurons fire action potentials at the same time. You
% will be able to answer this question by constructing a PSTH. Also you
% will be able to tell whether one tends to fire before or after the other.
% The PSTH has some conceptual similarity to the spike-triggered average if
% you are familiar with this technique. The idea in both cases is to use
% events in one recording as time points to observe events in another
% recording. In the spike-triggered average, action potentials in one
% recording are used to trigger sweeps of data from the other, and the
% averaged sweeps will show the relationship between firing events in both
% sequences.

% data is already time aligned since taken from 300ms before arm movement
% onset 
% want to look at individual neuron responses to a single stimulus, across
% different trials 

% Plot all neurons averaged over all trials for each direction to visualise
% neuronal direction preference to angles (directions 1 - 8) 
noBins = 30;
noTrials = 100;
all_nrns = time_averaged_neurons(noTrials);

figure
for direction_no = [1:1:8]
    leg = [];
    subplot(4, 2, direction_no)
    avg_fr = [];
    for neuron_no = [1:1:98]
        [avg_fr] = peri_stimulus_time_histogram(all_nrns, neuron_no, direction_no, noBins);
        if max(avg_fr > 0.005) % only show legend for neurons with preference - 0.01 arbitrarily chosen 
            leg = [leg; plot(avg_fr, 'DisplayName', "Neuron " + neuron_no)];
%         plot(avg_fr)
        end
        hold on
    end        
    hold off
    legend(leg, 'Location', 'eastoutside')
    xlabel('Time (ms) From Commencement of Movement')
    ylabel('PDF')
    title("PDF for firing rate for direction " + direction_no + " Averaged over all Trials for each Neuron")
end

%% Plot hand positions for different trials
% figure
% c = linspace(1,10,length(trial(90, 1).handPos(1, :)));
% scatter3(trial(90, 1).handPos(1,:), trial(90, 1).handPos(2,:),trial(90, 1).handPos(3,:),[],c)
% figure 
% c = linspace(1,10,length(trial(90, 1).handPos(1, :)));
% scatter(trial(90, 1).handPos(1, :), trial(90, 1).handPos(2, :),[], c)
figure 
plot(trial(90, 1).handPos(1, 301:end-100))
hold on 
plot(trial(90, 1).handPos(2, 301:end-100))
plot(trial(90, 1).handPos(3, 301:end-100))

%% Tuning Curves 
% Approach 1: neurons coding for movement in certain directions, not
% necessarily angles??
figure 
for direction_no = [1:1:8]
     x1 = trial(1, direction_no).handPos(1,:); 
     x2 = trial(1, direction_no).handPos(2,:); 
     x3 = trial(1, direction_no).handPos(3,:); 
        for i = [2:1:noTrials] % start at i = 2 since i = 1 was used to initialise array x
            y1 = trial(i, direction_no).handPos(1,:);
            y2 = trial(i, direction_no).handPos(2,:);
            y3 = trial(i, direction_no).handPos(3,:);
            if size(x1, 2) > size(y1, 2)
                y1(numel(x1)) = 0; % fills 'missing' elements in y with 0 if y shorter than x
            elseif size(y1, 2) > size(x1, 2)
                x1(numel(y1)) = 0; % fills 'missing' elements in x with 0 if x shorter than y
            end
            if size(x2, 2) > size(y2, 2)
                y2(numel(x2)) = 0; % fills 'missing' elements in y with 0 if y shorter than x
            elseif size(y2, 2) > size(x2, 2)
                x2(numel(y2)) = 0; % fills 'missing' elements in x with 0 if x shorter than y
            end
            if size(x3, 2) > size(y3, 2)
                y3(numel(x3)) = 0; % fills 'missing' elements in y with 0 if y shorter than x
            elseif size(y3, 2) > size(x3, 2)
                x3(numel(y3)) = 0; % fills 'missing' elements in x with 0 if x shorter than y
            end

            x1 = x1 + y1;
            x2 = x2 + y2;
            x3 = x3 + y3;

        end
        time_array = [1:1:size(x1, 2)];   % array of time points in ms
        subplot(4, 2, direction_no)
        plot(time_array, x1/size(x1,2))
        hold on
        plot(time_array, x2/size(x2, 2))
        plot(time_array, x3/size(x3,2))
        xlabel('Time (ms)')
        ylabel('Position') 
        title("Average Hand Position over all Trials, Direction " + direction_no)
end
%%
all_nrns = time_averaged_neurons(100);

% %%
%     figure
%     neuron_no = 49;
%     for direction_no = [1:1:8]
%         subplot(4, 2, direction_no)
%         plot(x1/size(x1,2), mean(all_nrns(((direction_no-1)*98)+neuron_no,1:length(x1))))
% %         hold on 
% %         plot(x2/size(x2, 2), all_nrns(((direction_no-1)*98)+neuron_no,1:length(x2))/length(x2))
% %         plot(x3/size(x3,2), all_nrns(((direction_no-1)*98)+neuron_no,1:length(x3))/length(x3))
%         title("Tuning Curve: Neuron " + neuron_no + " Direction " + direction_no)
%         ylabel('Average Firing Rate')
%         xlabel('Movement Direction')
%         hold off
%     end
% %     plot(x1/size(x1,2), avg_fr)
% %     hold on 
% %     plot(x2/size(x2,2), avg_fr)
% %     plot(x3/size(x3,2), avg_fr)
% 
 
%% dfs Approach 2: neurons coding for movement in the angles  
for neuron_no = [1 10 20 40 50 60 80 90]
    tune = zeros(8,1);
    for direction_no = [1:1:8]
        tune(direction_no) = mean(all_nrns(((direction_no-1)*98)+neuron_no));
    end
    figure
    errorbar(tune)
end