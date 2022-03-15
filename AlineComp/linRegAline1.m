%% Linear Regression with instantaneous spike rates and velocity

clear
close all


data = load('monkeydata_training.mat');
trial = data.trial;

training_trials = trial(1:20, :);
[N_trials, N_angles] = size(training_trials);
N_neurons = size(training_trials(1,1).spikes, 1);

windowsize = 30;
[velx, vely, velz] = getvel2(training_trials, windowsize);

spike_rates = get_spike_rates(training_trials, windowsize);

spike_rates2 = get_spike_rates2(training_trials, windowsize);