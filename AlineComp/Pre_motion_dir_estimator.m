%% BMI Coursework
%% March 2022
%% Aline Buat

clear;
close all;


%% Init
load('monkeydata_training.mat');
N_trials = size(trial, 1);
N_reaching_angles = size(trial, 2);
N_neuralunits = size(trial(1,1).spikes, 1);


%% tuning curve
% Angle from 1st 300 ms


n_unit = 70;
N_bins = 25;


k_list = 1:N_reaching_angles;
color_list = {'#0072BD', '[0.8500 0.3250 0.0980]','[0.4940 0.1840 0.5560]',...
'[0.9290 0.6940 0.1250]','r','g','b','m'};

neuron_tuning = zeros(1, N_reaching_angles);
figure; hold on;
xlabel('t [ms]'); ylabel('Firing rate [Hz]');



for k = k_list
    [neuron_tuning_dir_k_mean, neuron_tuning_dir_k_std] = neuron_tuning_dir_k(trial, n_unit, k, N_bins);
    neuron_tuning(k) = neuron_tuning_dir_k_mean;
    neuron_tuning_std(k) = neuron_tuning_dir_k_std;
    fprintf("Dir %g: av spike rate=%g \n", k, neuron_tuning_dir_k_mean);

end
legend;

theta = (30+40*k_list)/180*pi;
figure;
polarplot([theta,theta(1)], [neuron_tuning,neuron_tuning(1)],'-.o', 'MarkerSize', 8);
title("Tuning Curve of Neuron unit ="+n_unit);


figure;grid on; hold on;
errorbar(k_list, neuron_tuning, neuron_tuning_std);
xlabel('k');
ylabel('Average firing rate');
title("Neuron unit ="+n_unit);

f_tuning = fit(k_list', neuron_tuning', 'gauss1');
plot(f_tuning);
f_smooth = smoothdata(neuron_tuning, 'gaussian', 4);
plot(f_smooth);


pdf_tuning = neuron_tuning/sum(neuron_tuning, 'all');
figure;
plot(pdf_tuning);
ylim([0,inf]);


%% Encode Population tuning

k_list = 1:N_reaching_angles;
n_list = 1:N_trials;


neuron_tuning = zeros(N_trials, N_reaching_angles);




