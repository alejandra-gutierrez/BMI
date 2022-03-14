%% 
training_data = load('monkeydata_training.mat')
training_data = training_data.trial
window_size = 20 ;
  
% Calc of velocity from position data. Change window size to reflect the length of interest
[x_vel_points, y_vel_points,spike_train_binned]= getvel(training_data, window_size) ;

[trials,angle]=size(training_data);

spikes = [];
direction = [];

neurons=length(training_data(1,1).spikes(:,1));
spike_angle = zeros(trials,neurons);

for a = 1:angle
    for n = 1:neurons
        for t = 1:trials
                spikesnr = sum(training_data(t,a).spikes(n,1:320));
                spike_angle(t,n) = spikesnr;
        end
    end
    spikes = [spikes; spike_angle];
    angles(1:trials) = a;
    direction = [direction, angles];
end


%%
knn = fitcknn(spikes,direction, 'NumNeighbors',8,'Standardize',1);
modelParameters.knn=knn;


%%
gscatter(spikes(:,:),direction,direction);







