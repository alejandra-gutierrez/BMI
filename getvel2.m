function [vel] = getvel2(trials , windowsize, t_step, t_start)

% x_vel: [max_t x N_trials x N_angles x Axis]

% no window overlap version 

    if ~exist('t_step', 'var') || isempty(t_step)
            t_step = 1;
    elseif t_step <=0
        t_step = ceil(windowsize/2);
    end
    if ~exist('t_start', 'var') || isempty(t_start) || t_start<1
        t_start = 1;
    end
      
    t_start = t_start - 1; % offset for iteration
    
    % make sure these are integers!
    t_start = floor(max([t_start, 0]));
    windowsize = ceil(windowsize) ;
    t_step = round(max([t_step, 1]));
    
    [N_trials, N_angles] = size(trials);    
    
    vel = zeros(N_trials, N_angles, 2000, 3); % make it bigger than estimated max size
    
    max_t=0;
    for n = 1:N_trials
        for k = 1:N_angles
            handPos = trials(n,k).handPos;
            timesteps = size(handPos, 2);
            if timesteps>max_t
                max_t = timesteps;
            end
            for t = t_start+windowsize:windowsize:timesteps
                vel(n, k, ceil((t-t_start)/t_step), 1) = (handPos(1,t)- handPos(1,t-windowsize+1))/windowsize;
                vel(n, k, ceil((t-t_start)/t_step), 2) = (handPos(2,t)- handPos(2,t-windowsize+1))/windowsize;
                vel(n, k, ceil((t-t_start)/t_step), 3) = (handPos(3,t)- handPos(3,t-windowsize+1))/windowsize;
                
            end
        end
    
    end
    
    
    % crop extra zeros at the end
    vel(:,:,max_t+1:end,:) = [];




end