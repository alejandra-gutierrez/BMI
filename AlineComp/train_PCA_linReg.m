%% Train PCA and linear Regression
%% Aline BUAT
% 17/03/2022

clear
close all

%% open file and init variables
data = load('monkeydata_training.mat');
trial = data.trial;

trials_tr = trial(1:80, :);
trials_test = trial(81:end, :);

[N_trials, N_angles] = size(trial);
N_neurons = size(trial(1,1).spikes, 1);
N_trials_tr = size(trials_tr, 1);
N_trials_test = size(trials_test, 1);

%%

% select direction for training
dir = 3;
windowsize = 26;


%% Get velocities and rates
tic;

fprintf("Founding spike rates and velocitites...")
% get hand Velocities
[velx, vely, velz] = getvel2(trials_tr, windowsize);

% spike_rate format [n x k] cell with [N_neurons x t_max_each]
spike_rate = get_spike_rates2(trials_tr, windowsize);
% spike_rate = get_spike_rates2(trials_tr, windowsize);

fprintf("done...")
toc;
%% Make average

t_mvmt_start = 290;
% average is format [N_neurons x t_max_all]
% spike_rate_av_trials = make_av_spike_rate(spike_rate, dir);
spike_rate_av_trials = make_av_spike_rate(spike_rate);

% extract principal components from average:
[principal_spikes, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials(:, t_mvmt_start:end), 0.01);

V_red = Vs(:, 1:M);

figure; plot(principal_spikes');
fprintf("Extracted principal component vectors...")
toc;
%% Extract principal components in training data and reduce time res (data size)
% extract principal components for data points

% reduce size of arrays by reducing time resolution (otherwise too big data
% for linear Regression operation)

% since rate/ velocity data is created with rolling difference average
% there is high correlation between close data points in time
spike_rate_k = spike_rate(:, dir);
% spike_rate_k = spike_rate;

for k_it = 1:N_angles
    for n = 1:N_trials_tr
        spikes_mean = mean(spike_rate{n, k_it}, 2);
        principal_spikes_tr{n, k_it} = V_red'*(spike_rate{n, k_it} - spikes_mean);
        principal_spikes_tr{n, k_it} = principal_spikes_tr{n, k_it}(:, 1:round(windowsize/2):end); % reduce time res in rates
    end
end


% reduce res in velx, vely, velz
velx = velx(:, :, 1:round(windowsize/2):end);
vely = vely(:, :, 1:round(windowsize/2):end);
velz = velz(:, :, 1:round(windowsize/2):end);


fprintf("Found principal components of data...")
toc;
%% Do linear Regression if possible
PCA_components_weights_x = linearRegression2(principal_spikes_tr, velx, dir)
PCA_components_weights_y = linearRegression2(principal_spikes_tr, vely, dir)
% PCA_components_weights_x = linearRegression2(principal_spikes_tr, velx)
% PCA_components_weights_y = linearRegression2(principal_spikes_tr, vely)
fprintf("Finished model training for k=%g ...", dir);
toc;
