%% Try to extract PCA from neurons
clear


data = load('monkeydata_training.mat');
trial = data.trial;


[N_trials, N_angles] = size(trial);
N_neurons = size(trial(1,1).spikes, 1);

%% Extract spike rates
trial_n = trial(1,1);

k=1;
spikes = trial_n.spikes;
spike_rate = get_spike_rates2(trial(:,k),30);
% spike_rate = spike_rate{1, 1};

spike_rate_av_trials = make_av_spike_rate(spike_rate, 1);


%%
spike_rate = spike_rate_av_trials;

spikes_bar = mean(spike_rate, 2);

A = cov(spike_rate'); % 98x98 covariance matrix
[V,D] = eig(A);
[d,ind] = sort(diag(D), 'descend');
Ds = D(ind, ind);
Vs = V(:, ind);

figure; plot(diag(Ds));

th_percent = 12;
Is = find(diag(Ds)<max(Ds,[],'all')/th_percent);
M = Is(1);
V_red = Vs(:,1:M); % principal component vectors


% dimentionality reduction by using M-dim representation of the L=98-dim
% input

% a is low-dimentional and decorrelated
a = V_red'*(spike_rate - spikes_bar);
figure;
plot(a');

%% training
[velx, vely, velz] = getvel2(trial(1:60,1), 20);
k=1;

rates = get_spike_rates2(trial(1:60,1), 30);

linearRegression2(rates, velx, 1, 1)

%%
input_rate =  get_spike_rates2(trial(1,k),30);
input_rate = input_rate{1};

% a is low-dimentional and decorrelated
a = V_red'*(input_rate - spikes_bar);
plot(a');



%% Reconstitute original (reduced) input x
x=[];
for m = 1:M
    x=x+a(m,:)*V_red(:,m);
end


