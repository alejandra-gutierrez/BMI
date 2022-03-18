function [x_vel, y_vel, z_vel] = getvel2(trials , windowsize)

% x_vel: [max_t x N_trials x N_angles]


[N_trials, N_angles] = size(trials);
N_neurons = size(trials(1,1).spikes, 1);


x_vel = zeros(N_trials, N_angles, 2000); % make it bigger than estimated max size
y_vel = zeros(N_trials, N_angles, 2000);
z_vel = zeros(N_trials, N_angles, 2000);

max_t=0;
for n = 1:N_trials
    for k = 1:N_angles
        handPos = trials(n,k).handPos;
        timesteps = size(handPos, 2);
        if timesteps>max_t
            max_t = timesteps;
        end
        for t = 1+windowsize:1:timesteps
            x_vel(n, k, t) = handPos(1,t)- handPos(1,t-windowsize+1);
            y_vel(n, k, t) = handPos(2,t)- handPos(2,t-windowsize+1);
            z_vel(n, k, t) = handPos(3,t)- handPos(3,t-windowsize+1);
        end
    end

end


% crop extra zeros at the end
% x_vel(max_t+1:end,:,:) = [];
% y_vel(max_t+1:end,:,:) = [];
% z_vel(max_t+1:end,:,:) = [];
x_vel(:,:,max_t+1:end) = [];
y_vel(:,:,max_t+1:end) = [];
z_vel(:,:,max_t+1:end) = [];



end