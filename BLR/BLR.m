%---------------------------------------- train sample  -----------------------------------------%
%  Funtion: ��ƫ����sum1-sum2-a*theta,����sum2��һ��'����'-1*2;����������theta�й�
%  alpha = 0.0001�Ǻ��ʵģ���Ҫ��ƫ����sum2�ǳ���������10^4�����������ȡ0.01ÿ�α仯�ܴ�ƫ���������䣬���ǲ�����0
%  ����ѧ�Ϸ�������ƫ��-derivativeΪ0(С��ĳ����eps),������һ��ʱ�䣬������Ϊ����ȡ��ֵ���������ƫ���������䣬��˵��ѧϰ��̫����
%  Convention: i means sample, j means feature
clc; clear; tic; format long;                                  % clc-clear Command Window, clear-clear Workspace, tic��toc���Ժ�ʱ����, ��long��ʽ��ʾ

train_file = xlsread('BLR_training_data.xls');                 % ����train����
train_data = train_file(:, 1:2);                               % ȡ��������ǩ-ǰ2��
[train_samples,features] = size(train_data);                   % ��ȡ�������ݵ�samples��features

% ��������
a = 1;                                                         % theta������̬�ֲ��Ĳ���                                     
alpha = 0.0002;                                                % ѧϰ��0.0001
theta = zeros(features, 1);                                    % theta������ͬά
theta(:,1) = [0.1; 0.2];                                       % ��ʼֵ

% sum2��theta�޹أ���ǰ����
% ɸѡ����
train_data_y0 = train_file(train_file(:, 3) == 0, :);         % train_data_y1�Ǳ�ǩΪ0���������ݣ�������ǩ
sum2 = sum(train_data_y0(:, 1:2));
clear train_file train_data_y0;

result = zeros(100, 3);     % ��������[ѭ������, ��Ӧ��\phi]
derivative = [1; 1];        % matlab û��do-whileѭ��������Ϊ1���������һ���ж�����
k = 1;                      % ��¼��������
while (norm(derivative, 1) > 10^(-3))     % ������ȡ1����
    disp(derivative);                     % �۲�ƫ��
    result(k,:)  = [k, theta'];           % ���������۲�ÿ�ε�phiȡֵ�仯
    k = k+1;                              % ��������
    
    % ��ȡL(theta)��theta��ƫ��
    sum1 = zeros(features,1);
    for i = 1:train_samples
        feature = train_data(i, :)';                             % feature
        sum1 = sum1 + feature/(1+exp(theta'*feature));          
    end
    derivative = sum1 - sum2' - a*theta;                         % ��ƫ��
    theta = theta + alpha*derivative;                            % ����theta
end
clear i derivative theta;

% ����theta�ı仯����
plot(result(:, 1), result(:, 2:3)); grid on;
title('\alpha=0.0002'); xlabel('��������k'), ylabel('\theta������'); 
set(gca,'ytick',-12:0.5:1);                   % ����y����ʾ����
