function [x_vel, y_vel, z_vel] = getvel2(trials , windowsize)

% x_vel: [max_t x N_trials x N_angles]


[N_trials, N_angles] = size(trials);
N_neurons = size(trials(1,1).spikes, 1);


x_vel = zeros(2000 ,N_trials, N_angles); % make it bigger than estimated max size
y_vel = zeros(2000, N_trials, N_angles);
z_vel = zeros(2000, N_trials, N_angles);

max_t=0;
for n = 1:N_trials
    for k = 1:N_angles
        handPos = trials(n,k).handPos;
        timesteps = size(handPos, 2);
        if timesteps>max_t
            max_t = timesteps;
        end
        for t = floor(1+windowsize/2:1:timesteps-windowsize/2)
            x_vel(t, n, k) = handPos(1,t+windowsize/2)- handPos(1,t-windowsize/2);
            y_vel(t, n, k) = handPos(2,t+windowsize/2)- handPos(2,t-windowsize/2);
            z_vel(t, n, k) = handPos(3,t+windowsize/2)- handPos(3,t-windowsize/2);
        end
    end

end


% crop extra zeros at the end
x_vel(max_t+1:end,:,:) = [];
y_vel(max_t+1:end,:,:) = [];
z_vel(max_t+1:end,:,:) = [];
% x_vel(:,:,max_t+1:end) = [];
% y_vel(:,:,max_t+1:end) = [];
% z_vel(:,:,max_t+1:end) = [];



end