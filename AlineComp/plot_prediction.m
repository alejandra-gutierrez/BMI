%% Plot hand position

disp_dir =7;
disp_n = 6;

[velx_test, vely_test, velz_test] = getvel2(trials_test, windowsize);

t_max = size(trials_test(disp_n, disp_dir).handPos, 2);
%% Plot velocity
figure;
subplot(1,2,1);
hold on
plot(squeeze(velx_test(disp_n,disp_dir,1:t_max)), 'DisplayName', 'Actual velx');
plot(squeeze(velx_estimated(disp_n,disp_dir,1:t_max)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('Velocity x');
title('Velocity estimation with PCA and linear Regression',"(with Prior dir k knowledge), k="+disp_dir);
lgd1 = legend; lgd1.Location='northwest';

subplot(1,2,2);
hold on
plot(squeeze(vely_test(disp_n, disp_dir,1:t_max)), 'DisplayName', 'Actual velx');
plot(squeeze(vely_estimated(disp_n, disp_dir,1:t_max)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('velocity y');
title('Velocity estimation with PCA and linear Regression',"(with Prior dir k knowledge), k="+disp_dir);
lgd1 = legend; lgd1.Location='northwest';

%% Plot handPos
figure; 
subplot(1,2,1);
hold on
plot(trials_test(disp_n, disp_dir).handPos(1,:), 'DisplayName', 'Actual handPosx');
plot(squeeze(handPos_estimated_x(disp_n, disp_dir, 1:t_max)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('Position x');
title('Position estimation with PCA and Linear Regression',"(with Prior dir k knowledge), k="+disp_dir);
lgd2 = legend; lgd2.Location = 'northwest';

subplot(1,2,2);
hold on
plot(trials_test(disp_n, disp_dir).handPos(2,:), 'DisplayName', 'Actual handPosy');
plot(squeeze(handPos_estimated_y(disp_n, disp_dir, 1:t_max)), 'DisplayName', 'Estimated');
xlabel('t (ms)');
ylabel('Position y');
title('Position estimation with PCA and Linear Regression',"(with Prior dir k knowledge), k="+disp_dir);
lgd2 = legend; lgd2.Location = 'northwest';
