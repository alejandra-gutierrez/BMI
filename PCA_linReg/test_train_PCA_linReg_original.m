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
k = 3;
windowsize = 26;


%% Get velocities and rates
tic;

% get hand Velocities
[velx, vely, velz] = getvel2(trials_tr, windowsize);

% spike_rate format [n x k] cell with [N_neurons x t_max_each]
spike_rate = get_spike_rates2(trials_tr, windowsize);
% spike_rate = get_spike_rates2(trials_tr, windowsize);

fprintf("Found spike rates and velocitites...")
toc;
%% Make average

t_mvmt_start = 290;
% average is format [N_neurons x t_max_all]
spike_rate_av_trials = make_av_spike_rate(spike_rate, k);

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
% spike_rate_k = spike_rate(:,k);
spike_rate_k = spike_rate;

for k_it = 1:N_angles
    for n = 1:N_trials_tr
        spikes_mean = mean(spike_rate{n,k_it}, 2);
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
PCA_components_weights_x = linearRegression2(principal_spikes_tr, velx, k)
PCA_components_weights_y = linearRegression2(principal_spikes_tr, vely, k)
fprintf("Finished model training for k=%g ...", k);
toc;


%% Get velocities and rates
% k = 1;
% get hand Velocities
[velx, vely, velz] = getvel2(trials_test, windowsize);

% spike_rate format [n x k] cell with [N_neurons x t_max_each]
spike_rate_test = get_spike_rates2(trials_test, windowsize);

%% Test model

velx_estimated = zeros(size(velx));
vely_estimated = zeros(size(vely));

t_max = size(velx_estimated, 3);
principal_spikes_test = cell(N_trials_test, N_angles);

for n = 1:N_trials_test
    spikes_mean = mean(spike_rate_test{n,1}, 2);
    principal_spikes_test{n, k} = V_red'*(spike_rate_test{n, k} - spikes_mean);
    principal_spikes_test{n, k}(M, t_max) = 0; % zero padding
    velx_estimated(n, k, :) = PCA_components_weights_x'*principal_spikes_test{n,k};
    vely_estimated(n, k, :) = PCA_components_weights_y'*principal_spikes_test{n,k};
end

fprintf("Done finding principal components of test spike rates.\n");

%% get hand position from velocity


handPos_estimated_x = zeros(size(velx_estimated));
handPos_estimated_y = zeros(size(vely_estimated));

for k_it = 1:N_angles
    for n = 1:N_trials_test
        % correct for initial hand position (known)
        handPos_estimated_x(n, k_it, :) = handPos_estimated_x(n, k_it, :) + trials_test(n, k_it).handPos(1, 1);
        handPos_estimated_y(n, k_it, :) = handPos_estimated_y(n, k_it, :) + trials_test(n, k_it).handPos(2, 1);
    
        for t = t_mvmt_start:size(velx_estimated, 3)
            handPos_estimated_x(n, k_it, t) = sum(velx_estimated(n, k_it, t_mvmt_start:t))/10;
            handPos_estimated_y(n, k_it, t) = sum(vely_estimated(n, k_it, t_mvmt_start:t))/10;
        end
    end
end




%% Plot velocity
figure;
subplot(1,2,1);
hold on
plot(squeeze(velx(3,k,:)), 'DisplayName', 'Actual velx');
plot(squeeze(velx_estimated(3,k,:)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('Velocity x');
title('Velocity estimation with PCA and linear Regression',"(with Prior dir k knowledge), k="+k);

subplot(1,2,2);
hold on
plot(squeeze(vely(3,k,:)), 'DisplayName', 'Actual velx');
plot(squeeze(vely_estimated(3,k,:)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('velocity y');
title('Velocity estimation with PCA and linear Regression',"(with Prior dir k knowledge), k="+k);


%% Plot hand position
n = 8;

figure; 
subplot(1,2,1);
hold on
plot(trials_test(n, k).handPos(1,:), 'DisplayName', 'Actual handPosx');
plot(squeeze(handPos_estimated_x(n, k, :)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('Position x');
title('Position estimation with PCA and Linear Regression',"(with Prior dir k knowledge), k="+k);
legend;

subplot(1,2,2);
hold on
plot(trials_test(n, k).handPos(2,:), 'DisplayName', 'Actual handPosy');
plot(squeeze(handPos_estimated_y(n, k, :)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('Position y');
title('Position estimation with PCA and Linear Regression',"(with Prior dir k knowledge), k="+k);
legend;