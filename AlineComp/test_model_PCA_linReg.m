%% Requires model values first!


%% Get velocities and rates
tic;

k = 3;
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
    spikes_mean = mean(spike_rate_test{n, k}, 2);
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

%% Evaluate error 