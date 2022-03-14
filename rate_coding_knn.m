%% 
clear
%%
training_data = load('monkeydata_training.mat')
trial = training_data.trial
window_size = 30;
  
% Calc of velocity from position data. Change window size to reflect the length of interest
[x_vel_points, y_vel_points,spike_train_binned]= getvel(trial, window_size) ;

%% 
[N_trials,N_angles] = size(trial);

spikes = [];
direction = [];

N_neurons=length(trial(1,1).spikes(:,1));
spike_angle = zeros(N_trials,N_neurons);


for k = 1:N_angles
    for neuron = 1:N_neurons
        for n = 1:N_trials
                spikesnr = sum(trial(n,k).spikes(neuron,1:320));
                spike_angle(n, neuron) = spikesnr;
        end
    end
    spikes = [spikes; spike_angle];
    angles(1:N_trials) = k;
    direction = [direction, angles];
end


%%
knn = fitcknn(spikes,direction, 'NumNeighbors',8,'Standardize',1);
modelParameters.knn=knn;


%%
figure;
gscatter(spikes(:,:), direction,direction);



