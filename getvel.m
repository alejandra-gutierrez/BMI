function [x_vel_points, y_vel_points, spike_train_binned]= getvel(TrainingData, window_size)



[N_trials,N_angles] = size(TrainingData);
x_vel_points = zeros(40000, 1, N_angles);
y_vel_points = zeros(40000, 1, N_angles);
N_neurons = length(TrainingData(1,1).spikes(:,1));
spike_train_binned = zeros(40000, N_neurons, N_angles);

valx = zeros(1,N_angles);
valy = zeros(1,N_angles);

counter2 = 0;

for n = 1:N_trials
    for k = 1:N_angles
        timesteps = length(TrainingData(n,k).handPos(1,:));
        for t = 320:window_size:timesteps-window_size
            counter2 = counter2 +1;
            
            valx(k) = TrainingData(n,k).handPos(1,window_size+t)-TrainingData(n,k).handPos(1,1+t);
            x_vel_points(counter2, k) = valx(k);
            valy(k) = TrainingData(n,k).handPos(2,window_size+t)-TrainingData(n,k).handPos(2,1+t);
            y_vel_points(counter2, k) = valy(k);
            val(1, :, k) = var(TrainingData(n,k).spikes(:,(1+t):(window_size+t)),1,2);
            spike_train_binned(counter2,:,k) = val(1,:,k);
            
        end
    end
end



x_vel_points(counter2:end,:)=[];
y_vel_points(counter2:end,:)=[];
spike_train_binned(counter2-1:end,:,:)=[];


end

