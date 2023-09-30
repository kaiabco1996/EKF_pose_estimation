%EE5111 AY2023/2024 Semester 1
%CA 1: Multi-Sensor Pose Estimation for Autonomous Vehicles
%Group 8 Simulation Code
%This program simulates a sensor fusion design proposal using a GPS and
%IMU to determine location and heading of a vehicle in a 2D plane. Extended
%Kalman Filter is used to estimate the true state values while gaussian noise is
%applied to the true state values to simulate sensor outputs. 

%% Initialization

%clear all variables first and close all opened windows
clc;
clear all;
close all;

%Set number of states here which will be used for CTRV model.
%x_pos= Position X
%y_pos= Position Y
%yaw_angle= Heading angle
%vel=Velocity at Heading Direction
%yaw_rate= Heading rate
num_states=5;

%Set number of measurements here. In the real world, tracking systems can only
%measure the following through an IMU and GPS:
%states:
%x_pos= Position X
%y_pos= Position Y
%vel=Velocity at Heading Direction
%yaw_rate= Heading rate
num_meas=4;

%Set frequency of EKF measurement here:
T_freq= 50;
%Set frequency of GPS here to simulate variation of device performance:
GPS_freq = 50;

%Sample times for EKF and GPS are set below:
T= 1/T_freq;
T_GPS=1/GPS_freq;
tt = 100; %total simulation time
t = 0: T: tt; %Set timestamp array

% Set initial error covariance P first:
P = 1000*eye(5,5); %We set a large uncertainty first assuming we don't know location of vehicle

%Set Process Noise Covariance Matrix Q
%Assign standard deviation values for each state variable here:
sd_GPS_Q = 0.5*8.8*(T^2); %Assume 8.8 m/s^2 is the max acceleration for the vehicle
sd_yaw_Q = 0.1*T; %Assume 0.1 rad/s as maximum turn rate for the vehicle
sd_velocity_Q = 8.8*T; %Assume 8.8m/s^2 as maximum acceleration of the vehicle
sd_yaw_rate_Q = 1*T; %Assume 1 rad/s^2 as the maximum yaw rate acceleration for the vehicle
%Set Process Noise Covariance Matrix Q below. Each element in the diagonal
%are the variance values obtained by squaring the std dev values
Q = diag([(sd_GPS_Q^2) (sd_GPS_Q^2) (sd_yaw_Q^2) (sd_velocity_Q^2) (sd_yaw_rate_Q^2)]); 

%Set Measurement Noise Covariance Matrix R
%Assign standard deviation values for each measurement here:
sd_GPS_R = 6; %std dev for GPS measurement
%sd_yaw_R = 2; %std dev for yaw measurement
sd_velocity_R = 1; %std dev for velocity measurement
sd_yaw_rate_R = 0.1; %std dev for yaw rate measurement
%Set Measurement Noise Covariance Matrix R below. Each element in the diagonal
%are the variance values obtained by squaring the std dev values
%R = diag([(sd_GPS_R^2) (sd_GPS_R^2) (sd_yaw_R^2) (sd_velocity_R^2) (sd_yaw_rate_R^2)]);
R = diag([(sd_GPS_R^2) (sd_GPS_R^2) (sd_velocity_R^2) (sd_yaw_rate_R^2)]);

%% Extraction of Raw GPS Data from file

%Read Sensors Raw data from CSV File:
Raw_Data=readmatrix('Sensors Raw Data.csv');
%Divide raw data to different columns and convert to a 1D Array
date=Raw_Data(:,1); date=reshape(date.', 1, []);
time=Raw_Data(:,2); time=reshape(time.', 1, []);
millis=Raw_Data(:,3); millis=reshape(millis.', 1, []);
ax=Raw_Data(:,4); ax=reshape(ax.', 1, []);
ay=Raw_Data(:,5); ay=reshape(ay.', 1, []);
az=Raw_Data(:,6); az=reshape(az.', 1, []);
rollrate=Raw_Data(:,7); rollrate=reshape(rollrate.', 1, []);
pitchrate=Raw_Data(:,8); pitchrate=reshape(pitchrate.', 1, []);
yawrate=Raw_Data(:,9); yawrate=reshape(yawrate.', 1, []);
oll=Raw_Data(:,10); oll=reshape(oll.', 1, []);
pitch=Raw_Data(:,11); pitch=reshape(pitch.', 1, []);
yaw=Raw_Data(:,12); yaw=reshape(yaw.', 1, []);
speed=Raw_Data(:,13); speed=reshape(speed.', 1, []);
course=Raw_Data(:,14); course=reshape(course.', 1, []);
latitude=Raw_Data(:,15); latitude=reshape(latitude.', 1, []);
longitude=Raw_Data(:,16); longitude=reshape(longitude.', 1, []);
altitude=Raw_Data(:,17); altitude=reshape(altitude.', 1, []);
pdop=Raw_Data(:,18); pdop=reshape(pdop.', 1, []);
hdop=Raw_Data(:,19); hdop=reshape(hdop.', 1, []);
vdop=Raw_Data(:,20); vdop=reshape(vdop.', 1, []);
epe=Raw_Data(:,21); epe=reshape(epe.', 1, []);
fix=Raw_Data(:,22); fix=reshape(fix.', 1, []);
satellites_view=Raw_Data(:,23); satellites_view=reshape(satellites_view.', 1, []);
satellites_used=Raw_Data(:,24); satellites_used=reshape(satellites_used.', 1, []);
temp=Raw_Data(:,25); temp=reshape(temp.', 1, []);
%For course (yaw) measurements, an angle of 0 degrees means the vehicle is
%travelling north while 90 degrees means it is travelling east. We need to
%offset the course values so that east direction becomes 0 degrees and
%north becomes 90 degrees.
course = (-course+90); 

%Next, we need to approximate longitude and latitude measurment of GPS to
%meters to check vehicle location
Radius_Earth = 6378388; %This is in meters
arc = 2*pi*(Radius_Earth+altitude)/360; %calculate in meters per degrees
dx = arc.*cos(latitude*(pi/180)).*[0 longitude(2:length(longitude))-longitude(1:length(longitude)-1)]; %We compute the x position in meters here
dy = arc.*[0 latitude(2:length(latitude))-latitude(1:length(latitude)-1)];%We compute the y position in meters here
mx = cumsum(dx); %Cumulative sum is performed so that each succeeding element represents the next x position in meters.
my = cumsum(dy); %Cumulative sum is performed so that each succeeding element represents the next y position in meters.
ds = sqrt(dx.^2+dy.^2); %calculate distance moved from previous location via distance formula
GPS_movement = ds ~= 0; %Flag array which will serve as trigger for EKF to indicate that vehicle has moved for each time step

%% EKF Setup Section

%Set measurement h matrix
h=eye(4,4);

% Set initial value of state variables here. The data from the 
x = [mx(1); my(1); course(1)*(pi/180); speed(1)/3.6+0.001; yawrate(1)*(pi/180)];

% Calculate U and V for quiver plots
%U = cos(x(3)) * x(4);
%V = sin(x(3)) * x(4);

%Setup true state matrix here:
true_states = [mx; my; speed/3.6; yawrate/180*pi];
sample_size = size(true_states, 2);

%Setup pseudo measurements from sensors here by adding gaussian noise:
noise_var = 2; %You can vary noise variance here
mx_sensor = mx + noise_var * randn(size(mx));
my_sensor = my + noise_var * randn(size(my));
speed_sensor = speed + noise_var * randn(size(speed));
yawrate_sensor = yawrate + noise_var * randn(size(yawrate));

%Setup pseudo measurement matrix here:
measured_states = [mx_sensor; my_sensor; speed_sensor/3.6; yawrate_sensor/180*pi];

global x0 x1 x2 x3 x4 Zx Zy Px Py Pdx Pdy Pddx Pddy Kx Ky Kdx Kdy Kddx
x0 = [];
x1 = [];
x2 = [];
x3 = [];
x4 = [];
Zx = [];
Zy = [];
Px = [];
Py = [];
Pdx = [];
Pdy = [];
Pddx = [];
Pddy = [];
Kx = [];
Ky = [];
Kdx = [];
Kdy = [];
Kddx = [];
dstate = [];

%% Run EKF Algorithm

%The for-loop below performs the extended kalman filter algorithm for all
%pseudo measurement and true state data within the sample size
for k = 1:sample_size

    % Peform EKF prediction step first:
    %Calculate prediction values for each state variables using equations
    %in process matrix F:
    if abs(yawrate(k)) < 0.0001 % When vehicle is moving in a straight line, perform this
        x(1) = x(1) + x(4)*T * cos(x(3));
        x(2) = x(2) + x(4)*T * sin(x(3));
        x(3) = x(3);
        x(4) = x(4);
        x(5) = 0.0000001; % This is to avoid calculation issues with the jacobian of process matrix A by not dividing value by 0
        dstate(k) = 0;
    else % otherwise
        x(1) = x(1) + (x(4)/x(5)) * (sin(x(5)*T+x(3)) - sin(x(3)));
        x(2) = x(2) + (x(4)/x(5)) * (-cos(x(5)*T+x(3)) + cos(x(3)));
        x(3) = mod((x(3) + x(5)*T + pi), (2.0*pi)) - pi;
        x(4) = x(4);
        x(5) = x(5);
        dstate(k) = 1;
    end
    
    % Calculate the Jacobian matrix JF of the process matrix F
    J13 = (x(4)/x(5)) * (cos(x(5)*T+x(3)) - cos(x(3)));
    J14 = (1.0/x(5)) * (sin(x(5)*T+x(3)) - sin(x(3)));
    J15 = (T*x(4)/x(5))*cos(x(5)*T+x(3)) - (x(4)/x(5)^2)*(sin(x(5)*T+x(3)) - sin(x(3)));
    J23 = (x(4)/x(5)) * (sin(x(5)*T+x(3)) - sin(x(3)));
    J24 = (1.0/x(5)) * (-cos(x(5)*T+x(3)) + cos(x(3)));
    J25 = (T*x(4)/x(5))*sin(x(5)*T+x(3)) - (x(4)/x(5)^2)*(-cos(x(5)*T+x(3)) + cos(x(3)));
    
    JF = [1.0, 0.0, J13, J14, J15;
          0.0, 1.0, J23, J24, J25;
          0.0, 0.0, 1.0, 0.0, T;
          0.0, 0.0, 0.0, 1.0, 0.0;
          0.0, 0.0, 0.0, 0.0, 1.0];
    
    % Compute the Covariance Matrix here during prediction step:
    P = JF * P * JF' + Q;

    %Next, we perform the estimation step of the EKF algorithm here:

    %Place all predicted state values from the prediction step in array form:
    x_pred = [x(1); x(2); x(4); x(5)];

    if GPS_movement(k) %If there is detected movement based from GPS data, use the the value below as the jacobian of the measurement matrix G 
        JG = [  1.0, 0.0, 0.0, 0.0, 0.0;
                0.0, 1.0, 0.0, 0.0, 0.0;
                0.0, 0.0, 0.0, 1.0, 0.0;
                0.0, 0.0, 0.0, 0.0, 1.0];
    else %else if there is no detected GPS movement, use the value below for the jacobian of the measurement matrix. This means there is no measurment value for mx and my.
        JG = [  0.0, 0.0, 0.0, 0.0, 0.0;
                0.0, 0.0, 0.0, 0.0, 0.0;
                0.0, 0.0, 0.0, 1.0, 0.0;
                0.0, 0.0, 0.0, 0.0, 1.0];
    end
    
    %Compute the kalman gain to be used for the current timestep
    S = JG * P * JG' + R;
    K = (P * JG') * inv(S);

    % The equations below will calculate estimated state values for current
    % timestep
    Z = measured_states(:, k); %Extract measurement values for current timestep
    y = Z - x_pred; % This is the Innovation or Residual step where measurment is updated (prediction + corrected value) weighted by the Kalman Gain.
    x = x + K * y; %This equation computes for the new estimated state variables at the current timestep

    % Update the covariance matrix in the estimation step:
    P = (eye(size(K, 1)) - K * JG) * P;

    %save states
    savestates(x, Z, P, K)
end

%% Plot filter performance

%Plot Covariance values for each estimated states
plotP(Px, Py, Pdx, Pdy, Pddx)


% Plot EKF Filter performance using a binary color map
figure('Position', [100, 100, 600, 600]);
im = imagesc(P);
colormap('gray'); % This is equivalent to the 'binary' colormap in Python
title(sprintf('Covariance Matrix P (after %i Filter Steps)', sample_size));
yticks(0:5);
yticklabels({'x', 'y', '\psi', 'v', '\psi dot'});
ax = gca; %returns current axes of the figure
ax.YAxis.FontSize = 22;
xticks(0:5);
xticklabels({'x', 'y', '\psi', 'v', '\psi dot'});
ax.XAxis.FontSize = 22;
xlim([-0.5,4.5]);
ylim([-0.5, 4.5]);
set(gca, 'YDir', 'reverse');
colorbar;
ax.Position = ax.Position + [0 0 -0.05 0]; % Adjust the position of the axis to make room for the colorbar

% Save figure
saveas(gcf, 'filter_performance.png');

%% Plot Kalman Gains

figure('Position', [10, 10, 1600, 900]);
%numSteps = length(measured_states(1,:));
numSteps = sample_size;

stairs(1:numSteps, Kx, 'DisplayName', 'x');
hold on;
stairs(1:numSteps, Ky, 'DisplayName', 'y');
stairs(1:numSteps, Kdx, 'DisplayName', '\psi');
stairs(1:numSteps, Kdy, 'DisplayName', 'v');
stairs(1:numSteps, Kddx, 'DisplayName', '\psi dot');

xlabel('Filter Step');
ylabel('');
title('Kalman Gain (the lower, the more the measurement fullfill the prediction)');
legend('Location','best','FontSize',18);
ylim([-0.1, 0.1]);

% Save figure
saveas(gcf, 'kalman_gains.png');

hold off;

%% Plot state vectors

figure('Position', [10, 10, 1600, 1600]);

% First subplot for x and y position
subplot(4,1,1);
stairs(1:numSteps, x0 - mx(1), 'DisplayName', 'x');
hold on;
stairs(1:numSteps, x1 - my(1), 'DisplayName', 'y');
title('Extended Kalman Filter State Estimates (State Vector x)');
legend('Location', 'best', 'FontSize', 22);
ylabel('Position (relative to start) [m]');
hold off;

% Second subplot
subplot(4,1,2);
stairs(1:length(measured_states(1,:)), x2, 'DisplayName', '\psi');
hold on;
stairs(1:length(measured_states(1,:)), mod((course/180*pi + pi), (2*pi)) - pi, 'DisplayName', '\psi (from GPS as reference)');
ylabel('Course');
legend('Location', 'best', 'FontSize', 16);
hold off;

% Third subplot
subplot(4,1,3);
stairs(1:length(measured_states(1,:)), x3, 'DisplayName', 'v');
hold on;
stairs(1:length(measured_states(1,:)), speed/3.6, 'DisplayName', 'v (from GPS as reference)');
ylabel('Velocity');
ylim([0, 30]);
legend('Location', 'best', 'FontSize', 16);
hold off;

% Fourth subplot
subplot(4,1,4);
stairs(1:length(measured_states(1,:)), x4, 'DisplayName', '\psi dot');
hold on;
stairs(1:length(measured_states(1,:)), yawrate/180*pi, 'DisplayName', '\psi dot (from IMU as reference)');
ylabel('Yaw Rate');
ylim([-0.6, 0.6]);
legend('Location', 'best', 'FontSize', 16);
xlabel('Filter Step');
hold off;

% Save figure
saveas(gcf, 'state_estimates.png');

%% Plot Vehicle Location and Heading

% Plot Measured and Estimated Vehicle Position
figure('Position', [10, 10, 1600, 900]);
plot(x0, x1, 'k', 'LineWidth', 5, 'DisplayName', 'Estimated Position' , 'Color', 'blue');
hold on;
scatter(mx_sensor, my_sensor, 'DisplayName', 'Measured Position' , 'Color', 'green');
scatter(x0(1), x1(1), 60, 'g', 'filled', 'DisplayName', 'Start');
scatter(x0(end), x1(end), 60, 'r', 'filled', 'DisplayName', 'Goal');
xlabel('X [m]');
ylabel('Y [m]');
title('Vehicle Position- Measured vs Estimated');
legend('Estimated Position', 'Measured Position', 'Start', 'Goal');
axis equal;
hold off;

% save figure
saveas(gcf, 'Measured_vs_Estimated_Pos.png');

% Plot True and Estimated Vehicle Position
figure('Position', [10, 10, 1600, 900]);
plot(x0, x1, 'k', 'LineWidth', 5, 'DisplayName', 'Estimated Position' , 'Color', 'blue');
hold on;
plot(mx, my, 'k', 'LineWidth', 5, 'DisplayName', 'True Position' , 'Color', 'black');
scatter(x0(1), x1(1), 60, 'g', 'filled', 'DisplayName', 'Start');
scatter(x0(end), x1(end), 60, 'r', 'filled', 'DisplayName', 'Goal');
xlabel('X [m]');
ylabel('Y [m]');
title('Vehicle Position- Estimated vs True');
legend('Estimated Position', 'True Position', 'Start', 'Goal');
axis equal;
hold off;

% save figure
saveas(gcf, 'True_vs_Estimated_Pos.png');

%Plot Estimated and True Vehicle Heading
figure('Position', [10, 10, 1600, 900]);
%The quiver function below plots the orientation where the vehicle is heading in
%the 2D-map
quiver(x0(1:100:end), x1(1:100:end), cos(x2(1:100:end)), sin(x2(1:100:end)), 'Color', 'blue', 'AutoScale', 'on', 'AutoScaleFactor', 0.5, 'LineWidth', 0.5, 'DisplayName', 'Vehicle Heading- EKF');
hold on;
quiver(mx(1:100:end), my(1:100:end), cosd(course(1:100:end)), sind(course(1:100:end)), 'Color', 'black', 'AutoScale', 'on', 'AutoScaleFactor', 0.5, 'LineWidth', 0.5, 'DisplayName', 'Vehicle Heading- True');
scatter(x0(1), x1(1), 60, 'g', 'filled', 'DisplayName', 'Start');
scatter(x0(end), x1(end), 60, 'r', 'filled', 'DisplayName', 'Goal');
xlabel('X [m]');
ylabel('Y [m]');
title('Vehicle Heading- Estimated vs True');
legend('Estimated Heading', 'True Heading', 'Start', 'Goal');
axis equal;
hold off;

% save figure
saveas(gcf, 'Heading_map.png');


%% Functions

function plotP(Px, Py, Pdx, Pdy, Pddx)
    % Determine the number of steps
    m = length(Px);
    
    % Create a new figure with specified size
    figure('Position', [100, 100, 1600, 900]);
    
    % Plot the data using semilogy and step plot
    semilogy(0:m-1, Px, 'DisplayName', 'x');
    hold on;
    stairs(0:m-1, Py, 'DisplayName', 'y');
    stairs(0:m-1, Pdx, 'DisplayName', '\psi');
    stairs(0:m-1, Pdy, 'DisplayName', 'v');
    stairs(0:m-1, Pddx, 'DisplayName', '\psi dot');
    
    % Add labels and title to the plot
    xlabel('Filter Step');
    ylabel('');
    title('Uncertainty (Elements from Matrix P)');
    
    % Add legend to the plot
    legend('Location', 'best', 'FontSize', 22);
    hold off;
end

% Update states
function savestates(x, Z, P, K)
    global x0 x1 x2 x3 x4 Zx Zy Px Py Pdx Pdy Pddx Kx Ky Kdx Kdy Kddx

    x0(end+1) = x(1);
    x1(end+1) = x(2);
    x2(end+1) = x(3);
    x3(end+1) = x(4);
    x4(end+1) = x(5);
    Zx(end+1) = Z(1);
    Zy(end+1) = Z(2);
    Px(end+1) = P(1,1);
    Py(end+1) = P(2,2);
    Pdx(end+1) = P(3,3);
    Pdy(end+1) = P(4,4);
    Pddx(end+1) = P(5,5);
    Kx(end+1) = K(1,1);
    Ky(end+1) = K(2,1);
    Kdx(end+1) = K(3,1);
    Kdy(end+1) = K(4,1);
    Kddx(end+1) = K(5,1);
end



