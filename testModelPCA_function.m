function [handPos_estimated_x, handPos_estimated_y, velx_estimated, vely_estimated, errX, errY] = testModelPCA_function(trials_test, model, disp_dir, disp_n, windowsize, t_mvt)
%% test the model using the previously trained model
% model fields:
% model.PCAweightsX = PCA_components_weights_x;
% model.PCAweightsY = PCA_components_weights_y;
% model.M = M;
% model.V_red = V_red;
% model.dir = dir;

%% Get velocities and rates
tic;
[N_trials_test, N_angles] = size(trials_test);
N_neurons = size(trials_test(1).spikes, 1);

% get hand Velocities
[velx_test, vely_test, velz_test] = getvel2(trials_test, windowsize);

% spike_rate format [n x k] cell with [N_neurons x t_max_each]
spike_rate_test = get_spike_rates2(trials_test, windowsize);

%% Test model

velx_estimated = zeros(size(velx_test));
vely_estimated = zeros(size(vely_test));

t_max = size(velx_estimated, 3);
principal_spikes_test = cell(N_trials_test, N_angles);

for k_it = 1: size(spike_rate_test, 2)
    for n_it = 1:N_trials_test
        V_red = model(k_it).V_red;
        M = model(k_it).M;
        wX = model(k_it).PCAweightsX;
        wY = model(k_it).PCAweightsY;
%         principal_spikes_test{n, k_it} = extract_principal_components(spike_rate_test{n, k_it}, V_red);
        spikes_mean = mean(spike_rate_test{n_it, k_it}, 2);
        principal_spikes_test{n_it, k_it} = V_red'*(spike_rate_test{n_it, k_it} - spikes_mean);
        principal_spikes_test{n_it, k_it}(M, t_max) = 0; % zero padding
        velx_estimated(n_it, k_it, :) = wX'*principal_spikes_test{n_it, k_it};
        vely_estimated(n_it, k_it, :) = wY'*principal_spikes_test{n_it, k_it};
    end
end

fprintf("Done finding principal components of test spike rates.\n");
toc;

%% get hand position from velocity


handPos_estimated_x = zeros(size(velx_estimated));
handPos_estimated_y = zeros(size(vely_estimated));

for k_it = 1:N_angles
    for n_it = 1:N_trials_test
        for t = t_mvt:size(velx_estimated, 3)
            handPos_estimated_x(n_it, k_it, t) = sum(velx_estimated(n_it, k_it, t_mvt:t));
            handPos_estimated_y(n_it, k_it, t) = sum(vely_estimated(n_it, k_it, t_mvt:t));
        end
        % correct for initial hand position (known)
        handPos_estimated_x(n_it, k_it, :) = handPos_estimated_x(n_it, k_it, :) + trials_test(n_it, k_it).handPos(1, 1);
        handPos_estimated_y(n_it, k_it, :) = handPos_estimated_y(n_it, k_it, :) + trials_test(n_it, k_it).handPos(2, 1);
    end
end




%% Plot velocity
figure;
subplot(1,2,1);
hold on
plot(squeeze(velx_test(disp_n,disp_dir,:)), 'DisplayName', 'Actual velx');
plot(squeeze(velx_estimated(disp_n,disp_dir,:)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('Velocity x');
title('Velocity estimation with PCA and linear Regression',"(with Prior dir k knowledge), k="+disp_dir);
lgd = legend; lgd.Location='northwest';

subplot(1,2,2);
hold on
plot(squeeze(vely_test(disp_n, disp_dir,:)), 'DisplayName', 'Actual velx');
plot(squeeze(vely_estimated(disp_n, disp_dir,:)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('velocity y');
title('Velocity estimation with PCA and linear Regression',"(with Prior dir k knowledge), k="+disp_dir);
lgd = legend; lgd.Location='northwest';

%% Plot hand position

figure; 
subplot(1,2,1);
hold on
plot(trials_test(disp_n, disp_dir).handPos(1,:), 'DisplayName', 'Actual handPosx');
plot(squeeze(handPos_estimated_x(disp_n, disp_dir, :)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('Position x');
title('Position estimation with PCA and Linear Regression',"(with Prior dir k knowledge), k="+disp_dir);
lgd = legend; lgd.Location='northwest';

subplot(1,2,2);
hold on
plot(trials_test(disp_n, disp_dir).handPos(2,:), 'DisplayName', 'Actual handPosy');
plot(squeeze(handPos_estimated_y(disp_n, disp_dir, :)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('Position y');
title('Position estimation with PCA and Linear Regression',"(with Prior dir k knowledge), k="+disp_dir);
lgd = legend; lgd.Location='northwest';

%% Evaluate error

errX = zeros(size(handPos_estimated_x));
errY = zeros(size(handPos_estimated_y));

t_max_all = size(errX, 3);
for n_it = 1:N_trials_test
    for k_it = 1:N_angles
%       fprintf("Size of estimation mat: "); disp(size(handPos_estimated_y));
%       fprintf("Size of original mat: "); disp(size(trials_test(n_it, k_it).handPos));
        t_max = size(trials_test(n_it, k_it).handPos, 2);
        handPos_estimated_x(n_it, k_it, t_max+1:end) = 0; % remove offset after end of hand Recording
        handPos_estimated_y(n_it, k_it, t_max+1:end) = 0; % remove offset after end of hand Recording

        handPos_actualX = trials_test(n_it, k_it).handPos(1, :);
        handPos_actualX(t_max_all) = 0; % zero padding
        handPos_actualY = trials_test(n_it, k_it).handPos(1, :);
        handPos_actualY(t_max_all) = 0; % zero padding
        errX(n_it, k_it, :) = handPos_actualX - squeeze(handPos_estimated_x(n_it, k_it, :))';
        errY(n_it, k_it, :) = handPos_actualY - squeeze(handPos_estimated_y(n_it, k_it, :))';
    end
end


end