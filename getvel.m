function [x_vel_points, y_vel_points, spike_train_binned]= getvel(TrainingData, window_size)

    [trial,angle] = size(TrainingData);
     x_vel_points = zeros(40000, 1,angle);
     y_vel_points = zeros(40000, 1,angle);
     neurons=length(TrainingData(1,1).spikes(:,1));
     spike_train_binned = zeros(40000,neurons,angle);
     counter2=1;
    for i= 1:trial
        for j=1:angle
            timesteps=length(TrainingData(i,j).handPos(1,:));
        for t = 320:window_size:timesteps-window_size
            valx(:,j)=TrainingData(i,j).handPos(1,window_size+t)-TrainingData(i,j).handPos(1,1+t);
            x_vel_points(counter2,j) = valx(:,j);
            valy(:,j)=TrainingData(i,j).handPos(2,window_size+t)-TrainingData(i,j).handPos(2,1+t);
            y_vel_points(counter2,j) = valy(:,j);
            val(1,:,j)= var(TrainingData(i,j).spikes(:,(1+t):(window_size+t)),1,2);
            spike_train_binned(counter2,:,j) = val(1,:,j);
            counter2 = counter2 +1;
        end
        end
    end
    x_vel_points(counter2-1:end,:)=[];
    y_vel_points(counter2-1:end,:)=[];
    spike_train_binned(counter2-1:end,:,:)=[];
end

