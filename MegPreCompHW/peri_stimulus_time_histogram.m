function [time_array, avg_fr] = peri_stimulus_time_histogram(neuron_no, noTrials, direction_no, noBins)

    load('monkeydata_training.mat'); 
% ---------------------------------------------------------------------- %   
    % Construct array that is the sum of the spike data of all of the 
    % trials at a certain direction for a certain neuron  
    % trim to remove first 300 and last 100 as non-movement data
    
    x = trial(1, direction_no).spikes(neuron_no,301:end-100); 
    for i = [2:1:noTrials] % start at i = 2 since i = 1 was used to initialise array x
        y = trial(i, direction_no).spikes(neuron_no,301:end-100);
        if size(x, 2) > size(y, 2)
            y(numel(x)) = 0; % fills 'missing' elements in y with 0 if y shorter than x
        elseif size(y, 2) > size(x, 2)
            x(numel(y)) = 0; % fills 'missing' elements in x with 0 if x shorter than y
        end
        x = x + y;
    end
    time_array = [1:1:size(x, 2)];   % array of time points in ms 

% ---------------------------------------------------------------------- %   
    % Create dataset for histogram. Instead of one entry for each time
    % point containing the frequency (as in array x), time (ms) is appended
    % to matrix h the number of times spikes are seen across all trials
    
    % loop through array x. each element in x contains the number of spikes
    % that occured at that time point, across all trials for that neuron
    % (i.e. not yet averaged). time point in ms is given by index of array
    % i.e. element 50 corresponds to events at 50ms. 
    % if event at a time point occurred in any trial, contents of x is 
    % nonzero. time point is appended to array h the number of times at
    % which the event occured. i.e. if 7 trials had event at time point j,
    % that time point j ms is appended 7 times into h. this puts the data
    % into the correct format to use with histogram function. 
    
    h = [];
    for j = [1:1:size(x, 2)]
        if x(j) ~= 0
            counter = 1;
            while counter <= x(j)
                h = [h time_array(j)];
                counter = counter + 1;
            end
        end
    end
    
% % regular histogram plot
%     figure
%     histogram(h, noBins);
%     xlabel('Time (ms) from start of movement/showing stimulus')
%     ylabel("Firing rate (APs per " + length(x)/noBins + "ms")
    

% histfit plot
%     figure
%     histfit(h, noBins);
%     xlabel('Time (ms)')
%     ylabel("Firing rate (APs per " + length(x)/noBins + "ms")
%     pd1 = fitdist(transpose(h), 'Normal')
    
%     histogram(h, noBins)
%     title("PSTH for direction " + direction_no + ", neuron " + neuron_no) 
%     xlabel('Time (s)')
%     ylabel("Firing rate (APs per " + length(x)/noBins + "ms")
%     hold on
    if length(h) == 0
        h = zeros(1,length(x));
    end
    pd2 = fitdist(transpose(h), 'Kernel');
    avg_fr = pdf(pd2, time_array);
%     plot(time_array/1000, hist_data);
%     title("PDF for firing rate for direction " + direction_no + ", neuron " + neuron_no) 
%     xlabel('Time (s)')


end