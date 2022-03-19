%% Test input values
% PCA and linear regression
% go to plot_prediction to plot 

disp_dir =1 ;
disp_n = 1;

windowsize = 26;
t_mvt = 280;
t_train = tic;

run train_PCA_linReg.m
t_End_train = toc(t_train);
fprintf("\nEnd of training:  time = %g\n", t_End_train);


%%
% make all models correspond to undertermined dir model on purpose
% to test for what happens without prior knowledge
% for k_it = 1:N_angles
%     model(k_it) = model(N_angles+1);
% end
%%
t_Test = tic;
[handPos_estimated_x, handPos_estimated_y, velx_estimated, vely_estimated, errX, errY] = testModelPCA_function(trials_test, model, disp_dir, disp_n, windowsize, t_mvt);
t_End_test = toc(t_Test);
fprintf("\nEnd of testing:  time = %g\n", t_End_test);
%%
run plot_prediction.m

%%
save('estimations.mat','model', 'handPos_estimated_x', 'handPos_estimated_y', 'velx_estimated', 'vely_estimated', '-mat' );
