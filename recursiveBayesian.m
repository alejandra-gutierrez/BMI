% input_datax
% input_datay
% output_datax
% output_datay
%%

function [prior_prob, predicted_sr] = recursiveBayesianVel(input_datax, input_datay, output_datax, output_datay)

            min_y_pos = -200; % value for minimum y velocity (arbitrarily-ish chosen) - take min() for first trial, choose values fairly substantially past this
            min_x_pos = -200;
            max_y_pos = 200;
            max_x_pos = 200;  

    
    % for vel_dp = [0 1 2 3 4]; % possible number of decimal places to round velocity values to
    for vel_dp = [0] % possible number of decimal places to round velocity values to
        yEdge = [min_y_pos:1*10^-vel_dp:max_y_pos]; % create position bins 
        % find indices of which positions belong to which bin 
        for i = [1:1:length(yEdge)-1]
            ind = find(output_datay > yEdge(i) & output_datay <= yEdge(i+1));
            if isempty(ind)
                mean_sr(i) = 0;
                std_sr(i) = 0;
                err_sr(i) = 0;
            else
            % get the mean of the spike rate that corresponds to each bin 
                mean_sr(i) = mean(input_datay(ind));
            % standard deviation 
                std_sr(i) = std(input_datay(ind));
            % standard error
                err_sr(i) = std_sr(i)/sqrt(length(ind));
            end
        end
        yCenter = yEdge(1:end-1) + (1*10^-vel_dp)/2; % centre of each bin

        noNeurons = size(input_datay,1);
        for n = 1:1:noNeurons
            mdl(n,:) = glmfit(output_datay,input_datay(n,:), 'poisson'); % this is the correct way around - looking at encoding here rather than decoding, to predict firing rate rather than velocity/position
            predicted_sr(n,:)=exp(mdl(n,1) + mdl(n,2)*yCenter); % probability of spike rate given y velocity for neuron n 
        end

        % find p(vel) - prior probability
        prior_prob = histcounts(output_datay, [min_y_pos:1*10^-vel_dp:max_y_pos]);
        prior_prob = prior_prob./(sum(prior_prob)); % convert to probability
    end
    
end
