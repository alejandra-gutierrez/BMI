function hand_mvt_concat_dir_k = concat_hand_mvt_dir_k(trial, dir,n_list)
    % Concatenates spikes for a single neural unit from datafile trial
    % Given a movement direction k
    % neural unit neuron

    % output: 2D matrix A(n,t)
    % n is trial number
    % t is time (in ms)
    
    N_trials = size(trial, 1);
    N_reaching_angles = size(trial, 2);

    N_neuralunits = size(trial(1,1).handPos, 1);
    
    if ~exist('n_list', 'var') || isempty(n_list)
        n_list = 1:N_trials;
        fprintf("n_list undefined: new var length:%g\n", length(n_list));
    else
        fprintf("n_list already defined: length= %g\n",length(n_list))
    end   
    
    n=1;
    hand_mvt_concat_dir_k =trial(n, dir).handPos;

    for n = n_list(2:end)
        handPos = trial(n, dir).handPos;
        
        sf = size(hand_mvt_concat_dir_k) % size of old
        sn = size(handPos) % size of new
        if size(sf,3)<=1
            sf(3)=1;
        end
        if size(sn,3)<=1
            sn(3)=1;
        end
        
        if size(handPos, 2) ~= size(hand_mvt_concat_dir_k, 2)
            fprintf("Old [%g, %g, %g]",size(hand_mvt_concat_dir_k,1),size(hand_mvt_concat_dir_k,2),size(hand_mvt_concat_dir_k,3));
            fprintf("New [%g, %g, %g]\n",size(handPos,1),size(handPos,2),size(handPos,3));
            
            if size(handPos, 2) < size(hand_mvt_concat_dir_k, 2) % new smaller
                fprintf("New is smaller...");
                pad = ones(size(handPos, 1), size(hand_mvt_concat_dir_k,2)-size(handPos, 2), size(handPos,3)).*handPos(:,end);
                fprintf("pad size [%g, %g, %g]",size(pad,1),size(pad,2),size(pad,3));
                handPos = cat(2, handPos, pad);
                size(handPos)
                handPos(sn(1),sf(2)) = 0;
                size(handPos)
                fprintf("size chg...");

            elseif size(handPos, 2) > size(hand_mvt_concat_dir_k, 2) % new larger
                fprintf("Old is smaller...");
%                 pad = ones(size(hand_mvt_concat_dir_k,1), size(handPos, 2) - size(hand_mvt_concat_dir_k, 2), size(hand_mvt_concat_dir_k,3)).*hand_mvt_concat_dir_k(:,end,:);
%                 fprintf("pad size [%g, %g, %g]",size(pad,1),size(pad,2),size(pad,3));
% 
%                 hand_mvt_concat_dir_k = cat(2, hand_mvt_concat_dir_k, pad);
                hand_mvt_concat_dir_k(sf(1),sn(2),sf(3))=0;
                fprintf("size chg...");
            end
        end
        hand_mvt_concat_dir_k = cat(3, hand_mvt_concat_dir_k, handPos);
        fprintf("Size adjust successful.\n");
    end

     
    
end