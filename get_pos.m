function pos = get_pos(trials, windowsize, t_step, t_start)
    % output size positions: cell [N_trials x N_angles]
%       each is a double, size [N_neurons  x (t_max_each/t_step)]

% input trials - trial(n, k), handPos size(N_neurons, t_max)
%input t_step: not there = assume keep all time steps
% if t_step == 0 -> assume want default size reduction = windowsize/2


% no overlap in windows, windows have length windowsize

    [N_trials, N_angles]= size(trials);
    
    
    if ~exist('t_step', 'var') || isempty(t_step)
        t_step = 1;
    elseif t_step <=0
        t_step = ceil(windowsize/2);
    end
    if ~exist('t_start', 'var') || isempty(t_start)
        t_start = 1;
    end
    
    t_start = t_start - 1; % offset for iteration
    
    % make sure these are integers!
    t_start = floor(max([t_start, 0]));
    windowsize = ceil(windowsize) ;
    t_step = ceil(max([t_step, 1]));

    pos = zeros(N_trials, N_angles, 2000,3); % make it bigger than estimated max size

    max_t = 0;
    if rem(windowsize,2) == 0 % if even windowsize, take average position between two midpoints
        for n = 1:N_trials
            for k = 1:N_angles
                timesteps = size(trials(n,k).handPos(1, :), 2);
                if timesteps > max_t
                    max_t = timesteps;
                end
                for t = t_start+windowsize:windowsize:timesteps
                    pos(n, k, ceil((t-t_start)/t_step),1) = (trials(n,k).handPos(1, t-windowsize+t_step) + trials(n,k).handPos(1, t-windowsize+1+t_step))/2;
                    pos(n, k, ceil((t-t_start)/t_step),2) = (trials(n,k).handPos(2, t-windowsize+t_step) + trials(n,k).handPos(2, t-windowsize+1+t_step))/2;
                    pos(n, k, ceil((t-t_start)/t_step),3) = (trials(n,k).handPos(3, t-windowsize+t_step) + trials(n,k).handPos(3, t-windowsize+1+t_step))/2;
                end
            end
        end
    else    % if odd windowsize, take position at point in middle of window
        for n = 1:N_trials
            for k = 1:N_angles
                timesteps = size(trials(n,k).handPos(1, :), 2);
                if timesteps > max_t
                    max_t = timesteps;
                end
                for t = t_start+windowsize:windowsize:timesteps
                    pos(n, k, ceil((t-t_start)/t_step),1) = (trials(n,k).handPos(1, t-windowsize+t_step));
                    pos(n, k, ceil((t-t_start)/t_step),2) = (trials(n,k).handPos(2, t-windowsize+t_step));
                    pos(n, k, ceil((t-t_start)/t_step),3) = (trials(n,k).handPos(3, t-windowsize+t_step));
                end  
            end
        
        end
    end
    % crop extra zeros at the end

    pos(:,:,max_t+1:end,:) = [];

end

