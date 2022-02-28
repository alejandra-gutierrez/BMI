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
neuron_no = 58; 
noTrials = 100; % max no. trials is 100
singleNeuronManyTrials_Raster(neuron_no, noTrials, direction_no);

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
noBins = 100;
for direction_no = [1:1:8]
    figure
    time_array = [];
    avg_fr = [];
    for neuron_no = [1:1:98]
        [time_array, avg_fr] = peri_stimulus_time_histogram(neuron_no, noTrials, direction_no, noBins);
        plot(time_array, avg_fr);
        hold on
    end
    hold off
    xlabel('Time (ms) From Commencement of Movement')
    ylabel('PDF')
    title("PDF for firing rate for direction " + direction_no + "Averaged over all Trials for each Neuron")
end

%% Plot hand positions for different trials
figure
c = linspace(1,10,length(trial(90, 1).handPos(1, :)));
scatter3(trial(90, 1).handPos(1,:), trial(90, 1).handPos(2,:),trial(90, 1).handPos(3,:),[],c)
figure 
c = linspace(1,10,length(trial(90, 1).handPos(1, :)));
scatter(trial(90, 1).handPos(1, :), trial(90, 1).handPos(2, :),[], c)

%% Tuning Curves 
% Approach 1: neurons coding for movement in certain directions, not
% necessarily angles??

% Approach 2: neurons coding for movement in the angles 