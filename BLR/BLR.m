%---------------------------------------- train sample  -----------------------------------------%
%  Funtion: 求偏导：sum1-sum2-a*theta,其中sum2是一个'常数'-1*2;其他两项与theta有关
%  alpha = 0.0001是合适的，主要是偏导中sum2是常数而且是10^4数量级，如果取0.01每次变化很大，偏导正负跳变，就是不降到0
%  从数学上分析，当偏导-derivative为0(小于某个数eps),并持续一段时间，可以认为函数取极值。如果发现偏导正负跳变，则说明学习率太大了
%  Convention: i means sample, j means feature
clc; clear; tic; format long;                                  % clc-clear Command Window, clear-clear Workspace, tic与toc粗略耗时分析, 以long格式显示

train_file = xlsread('BLR_training_data.xls');                 % 读入train数据
train_data = train_file(:, 1:2);                               % 取出特征标签-前2列
[train_samples,features] = size(train_data);                   % 获取所有数据的samples，features

% 求后验概率
a = 1;                                                         % theta服从正态分布的参数                                     
alpha = 0.0002;                                                % 学习率0.0001
theta = zeros(features, 1);                                    % theta和样本同维
theta(:,1) = [0.1; 0.2];                                       % 初始值

% sum2与theta无关，提前计算
% 筛选数据
train_data_y0 = train_file(train_file(:, 3) == 0, :);         % train_data_y1是标签为0的所有数据，包括标签
sum2 = sum(train_data_y0(:, 1:2));
clear train_file train_data_y0;

result = zeros(100, 3);     % 输出结果，[循环次数, 对应的\phi]
derivative = [1; 1];        % matlab 没有do-while循环，设置为1，好满足第一次判断条件
k = 1;                      % 记录迭代次数
while (norm(derivative, 1) > 10^(-3))     % 列向量取1范数
    disp(derivative);                     % 观察偏导
    result(k,:)  = [k, theta'];           % 保存结果，观察每次的phi取值变化
    k = k+1;                              % 迭代次数
    
    % 求取L(theta)对theta的偏数
    sum1 = zeros(features,1);
    for i = 1:train_samples
        feature = train_data(i, :)';                             % feature
        sum1 = sum1 + feature/(1+exp(theta'*feature));          
    end
    derivative = sum1 - sum2' - a*theta;                         % 求偏导
    theta = theta + alpha*derivative;                            % 更新theta
end
clear i derivative theta;

% 绘制theta的变化曲线
plot(result(:, 1), result(:, 2:3)); grid on;
title('\alpha=0.0002'); xlabel('迭代次数k'), ylabel('\theta列向量'); 
set(gca,'ytick',-12:0.5:1);                   % 设置y轴显示精度
