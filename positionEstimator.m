function [x, y] = positionEstimator(test_data)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
  
  t = size(test_data, 2); % current timestamp in ms from length of data
  sc = 100; % distance scaling factor 
  label = randperm(8,1); % predicted direction (random number generator)

%   for i = [1:1:t]
%       switch label
%           case 1 % 30/180pi - 30deg
%               dv = [sqrt(3),1]; % direction vector for this direction 
%           case 2 % 70/180pi - 70deg
%               dv = [1,2.7474774194546];
%           case 3 % 110/180pi - 110deg
%               dv = [-1,2.7475];
%           case 4 % 150/180pi - 150deg 
%               dv = [-sqrt(3),1];
%           case 5 % 190/180pi - 190deg
%               dv = [-1,-0.17633];
%           case 6 % 230/180pi - 230deg
%               dv = [-1,-1.19175];
%           case 7 % 310/180pi - 310deg 
%               dv = [1, -1.1917536075574];
%           case 8 % 350/180pi - 350deg
%               dv = [1, -0.17632698790564];
%       end
% 
%       x(i) = (i/1000)*dv(1)*sc;
%       y(i) = (i/1000)*dv(2)*sc;
%   end
  
switch label
    case 1 % 30/180pi - 30deg
        dv = [sqrt(3),1]; % direction vector for this direction 
    case 2 % 70/180pi - 70deg
        dv = [1,2.7474774194546];
    case 3 % 110/180pi - 110deg
        dv = [-1,2.7475];
    case 4 % 150/180pi - 150deg 
        dv = [-sqrt(3),1];
    case 5 % 190/180pi - 190deg
        dv = [-1,-0.17633];
    case 6 % 230/180pi - 230deg
        dv = [-1,-1.19175];
    case 7 % 310/180pi - 310deg 
        dv = [1, -1.1917536075574];
    case 8 % 350/180pi - 350deg
        dv = [1, -0.17632698790564];
end

      x = (t/1000)*dv(1)*sc;
      y = (t/1000)*dv(2)*sc;
  
  
  
  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
   
end