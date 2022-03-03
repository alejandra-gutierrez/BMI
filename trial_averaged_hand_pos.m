function all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth)
    load('monkeydata_training.mat'); 
    all_psns = [];
    for direction_no = [1:1:8]
     x = trial(1, direction_no).handPos(axis,:); 
        for i = [2:1:noTrials] % start at i = 2 since i = 1 was used to initialise array x
            y = trial(i, direction_no).handPos(axis,:);
            if size(x, 2) > size(y, 2)
                y(numel(x)) = 0; % fills 'missing' elements in y with 0 if y shorter than x
            elseif size(y, 2) > size(x, 2)
                x(numel(y)) = 0; % fills 'missing' elements in x with 0 if x shorter than y
            end
            
            x = x + y;
            if mod(length(x), binWidth) ~= 0
                x = [x zeros(1, binWidth - mod(length(x), binWidth))];
            end
                r = reshape(x, binWidth, []);
            if size(r, 1) > 1
                r = r(1:end-binWidth+1,:);
            end
            all_psns(direction_no, 1:length(r)) = r/noTrials;
           
        end
    end

end
