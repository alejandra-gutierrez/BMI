%% Train PCA and linear Regression
%% Aline BUAT
% 17/03/2022



%% open file and init variables
data = load('monkeydata_training.mat');
trial = data.trial;

trials_tr = trial(10:81, :);
trials_test = trial([1:9,82:end], :);

[N_trials, N_angles] = size(trial);
N_neurons = size(trial(1,1).spikes, 1);
N_trials_tr = size(trials_tr, 1);
N_trials_test = size(trials_test, 1);

%%

% select direction for training
dir = 5;
windowsize = 26;
t_mvmt_start = 50;
t_step = windowsize/3;


%% Get velocities and rates
tic;

fprintf("Finding spike rates and velocitites...")
% get hand Velocities
[velx_tr, vely_tr, velz_tr] = getvel2(trials_tr, windowsize, t_step, t_mvmt_start);
fprintf("Velocities done...");
toc;
% spike_rate format [n x k] cell with [N_neurons x t_max_each]
% spike_rate = get_spike_rates2(trials_tr(:,dir), windowsize);
spike_rate = get_spike_rates2(trials_tr, windowsize, t_step, t_mvmt_start);

fprintf("Spike Rate done...")
toc;
%% Make average


fprintf("Extracting principal component vectors..."); tic;

% average is a double, size [N_neurons x t_max_all]


spike_rate_av_trials = make_av_spike_rate(spike_rate);
[principal_spikes, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials(:, :), 0.05);


for k_it=0:N_angles

        % extract principal components from average:
    spike_rate_av_trials = make_av_spike_rate(spike_rate, k_it); % if k_it = 0 (make it a sum of all components, all dir)
    [~, Vs, Ds, ~] = spikes_PCA(spike_rate_av_trials(:, :), 0.05);
    
   
    if (k_it ==0)
        k_it = N_angles+1; % just for indexing of models
    end
    %M = 5;
    model(k_it).dir = k_it;
    model(k_it).Vs = Vs;
    model(k_it).Ds = Ds;
    
    model(k_it).Vs = Vs;
    model(k_it).Ds = Ds;
    model(k_it).M = M;
    
    model(k_it).V_red = Vs(:, 1:M); % make them all the same size for training all simultaneously
    
end

%% Break the code and make model PCA all the same
% for k_it = 1:N_angles
%     model(k_it) = model(N_angles+1);
% end
%%
figure; plot(principal_spikes');
fprintf("done...")
toc;
%% Extract principal components in training data and reduce time res (data size)
% extract principal components for data points
fprintf("Extracting principal components of training data...\n"); tic;
% reduce size of arrays by reducing time resolution (otherwise too big data
% for linear Regression operation)

% since rate/ velocity data is created with rolling difference average
% there is high correlation between close data points in time
% spike_rate_k = spike_rate;
principal_spikes_tr = cell(N_trials_tr, N_angles);

for k_it = 1:size(spike_rate, 2)
    for n = 1:N_trials_tr
%         principal_spikes_tr{n, k_it} = extract_principal_components(spike_rate{n, k_it}, model(k_it).dir);
        V_red = model(k_it).V_red;
        spikes_mean = mean(spike_rate{n, k_it}, 2);
        principal_spikes_tr{n, k_it} = V_red'*(spike_rate{n, k_it} - spikes_mean);

        % remove 1st 290s from training points (before mvt) and reduce time res
%         principal_spikes_tr{n, k_it} = principal_spikes_tr{n, k_it}(:, t_mvmt_start:round(windowsize/2):end); 
        
        fprintf(".");
    end
    fprintf("\n");
end


% % reduce res in velx, vely, velz
% velx_tr = velx_tr(:, :, t_mvmt_start:round(windowsize/2):end);
% vely_tr = vely_tr(:, :, t_mvmt_start:round(windowsize/2):end);
% velz_tr = velz_tr(:, :, t_mvmt_start:round(windowsize/2):end);


fprintf("done...")
toc;
%% Do linear Regression

fprintf("Starting linear Regression..."); tic;

for k_it = 0:N_angles
    fprintf("Training model for k=%g ...\n", k_it);
    
    PCA_components_weights_x = linearRegression2(principal_spikes_tr, velx_tr, k_it)
    fprintf("Done with x_axis...\n");
    toc
    PCA_components_weights_y = linearRegression2(principal_spikes_tr, vely_tr, k_it)
    if (k_it == 0)
        k_it = N_angles+1; % just for indexing of models
    end
    model(k_it).PCAweightsX = PCA_components_weights_x;
    model(k_it).PCAweightsY = PCA_components_weights_y;
    % PCA_components_weights_y = linearRegression2(principal_spikes_tr, vely)
    fprintf("done with y_axis...\n");
    toc;
end



% model.Vs = Vs;
% model.Ds = Ds;
% model.PCAweightsX = PCA_components_weights_x;
% model.PCAweightsY = PCA_components_weights_y;
% model.M = M;
% model.V_red = V_red;
% model.dir = dir;
