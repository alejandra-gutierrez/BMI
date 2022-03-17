%% Test PCA and linear Regression
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
k = 1;
windowsize = 26;


%% Get velocities and rates

% get hand Velocities
[velx, vely, velz] = getvel2(trials_tr, windowsize);

% spike_rate format [n x k] cell with [N_neurons x t_max_each]
spike_rate = get_spike_rates2(trials_tr(:, k), windowsize);

%% Make average

% average is format [N_neurons x t_max_all]
spike_rate_av_trials = make_av_spike_rate(spike_rate, 1);

% extract principal components from average:
[principal_spikes, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, 0.05);

V_red = Vs(:, 1:M);

figure; plot(principal_spikes');


%% Extract principal components in training data and reduce time res (data size)
% extract principal components for data points

% reduce size of arrays by reducing time resolution (otherwise too big data
% for linear Regression operation)

% since rate/ velocity data is created with rolling difference average
% there is high correlation between close data points in time

for n = 1:N_trials_tr
    spikes_mean = mean(spike_rate{n,1}, 2);
    principal_spikes_tr{n, 1} = V_red'*(spike_rate{n, 1} - spikes_mean);
    principal_spikes_tr{n, 1} = principal_spikes_tr{n, 1}(:, 1:round(windowsize/2):end); % reduce time res in rates
end


% reduce res in velx, vely, velz
velx = velx(:, :, 1:round(windowsize/2):end);
vely = vely(:, :, 1:round(windowsize/2):end);
velz = velz(:, :, 1:round(windowsize/2):end);




%% Do linear Regression if possible
PCA_components_weights = linearRegression2(principal_spikes_tr, velx, 1, 1)


%% Test training

for n = 1:N_trials_test
    spikes_mean = mean(spike_rate{n,1}, 2);
    principal_spikes_tr{n, 1} = V_red'*(spike_rate{n, 1} - spikes_mean);
    principal_spikes_tr{n, 1} = principal_spikes_tr{n, 1}(:, 1:round(windowsize/2):end); % reduce time res in rates
end

%% Get velocities and rates

% get hand Velocities
[velx, vely, velz] = getvel2(trials_test, windowsize);

% spike_rate format [n x k] cell with [N_neurons x t_max_each]
spike_rate_test = get_spike_rates2(trials_test(:, k), windowsize);

velx_estimated = zeros(size(velx));
t_max = size(velx_estimated, 3);
for n = 1:N_trials_test
    spikes_mean = mean(spike_rate_test{n,1}, 2);
    principal_spikes_test{n, 1} = V_red'*(spike_rate_test{n, 1} - spikes_mean);
    principal_spikes_test{n,1}(M, t_max) = 0; % zero padding
    velx_estimated(n, 1,:) = PCA_components_weights'*principal_spikes_test{n,1};
end

figure; hold on
plot(squeeze(velx(1,1,:)), 'DisplayName', 'Actual velx');
plot(squeeze(velx_estimated(1,1,:)), 'DisplayName', 'Estimated');
xlabel('t');
ylabel('velocity x');
title('Velocity estimation with PCA and linear Regression','(with Prior dir k knowledge)');

%% get hand position from velocity

