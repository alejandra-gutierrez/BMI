% find 'average' hand trajectory for each direction 1 - 8 ]
% average taken across number of trials

function all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth)
    load('monkeydata_training.mat'); 
    all_psns = cell(1,8);
    for direction_no = [1:1:8]
     x = trial(1, direction_no).handPos(axis,:); 
        for i = [2:1:noTrials] % start at i = 2 since i = 1 was used to initialise array x
            y = trial(i, direction_no).handPos(axis,:);
            if size(x, 2) > size(y, 2)
                y(numel(x)) = 0; % fills 'missing' elements in y with 0 if y shorter than x
            elseif size(y, 2) > size(x, 2)
                x(numel(y)) = 0; % fills 'missing' elements in x with 0 if x shorter than y
            end
            
            x = x + y; % iterative sum of all trials for a direction 
        end
        
        
        if mod(length(x), binWidth) ~= 0 % if length of summed trials not a multiple of the chosen bin width, pad array
            x = [x NaN(1, binWidth - mod(length(x), binWidth))];
        end
        r = reshape(x, binWidth, []); % reshape array to have same number of rows as the bin width 
        
        % first row will have data points that correspond to the start of
        % the bins 
        if size(r, 1) > 1
            r = r(1:end-binWidth+1,:); % keep only the first row
        end
        
        all_psns{direction_no} = r/noTrials;
%         if length(r) > size(all_psns, 2)
%             diff = length(r) - size(all_psns, 2);
%             padding = NaN(size(all_psns, 1), diff); 
%             all_psns = [all_psns, padding];
%         elseif length(r) < size(all_psns, 2)
%             diff = size(all_psns, 2) - length(r);
%             padding = NaN(1, diff);
%             r = [r, padding];
%         end
%         all_psns(direction_no, 1:length(r)) = r/noTrials;
%             
    end

end


