%---------------------------------------- train sample -----------------------------------------%
%  Funtion: solve paramenter(general), include phi,[mu0-2*1,sigma0-2*2],[mu1-2*1,sigma1-2*2]
%  下面会定义几个矩阵，来存储这些值。如果是其他编程语言，就会定义相关的数据结构来表示，所以matlab还是方便的
%  Convention: i means sample, j means feature;这里二维正态分布，不妨记第一列数据为X，第二列数据为Y
clc; clear; tic;                                 % clc-clear Command Window, clear-clear Workspace, tic与toc粗略耗时分析            

train_file = xlsread('GDA_training_data.xls');   % 读入train数据    

% 筛选数据-也可以用find,由于标签是二值数据，所以省了find
train_data_y1 = train_file(train_file(:, 3) == 1, :);   % train_data_y1是标签为1的所有数据，包括标签
train_data_y0 = train_file(train_file(:, 3) == 0, :);   % train_data_y1是标签为0的所有数据，包括标签

mu = zeros(3, 1, 2);                                         % 产生两个矩阵-[mu1(X);mu1(Y)]和[mu0(X);mu0(Y)],最后一行是标签-可忽略
sigma = zeros(2, 2, 2);                                      % 产生两个矩阵-存储协方差
Num_totol = [length(train_data_y1); length(train_data_y0)];  % sample正负样本的个数，[1,1]-正，[2,1]-负

% 求解参数\theta
phi = Num_totol(1, 1)/sum(Num_totol);                               % phi
mu(:, 1, 1) = mean(train_data_y1)';                                 % y=1下的[mu1(X);mu1(Y)]
mu(:, 1, 2) = mean(train_data_y0)';                                 % y=0下的[mu0(X);mu0(Y)]
sigma(:, :, 1) =  cov(train_data_y1(:, 1), train_data_y1(:, 2));    % y=1下的sigma1
sigma(:, :, 2) =  cov(train_data_y0(:, 1), train_data_y0(:, 2));    % y=0下的sigma0

scatter(train_data_y1(:, 1), train_data_y1(:, 2), '.'); hold on;    % 画出散点图 y=1
scatter(train_data_y0(:, 1), train_data_y0(:, 2), '.', 'r');        % y=0
xlabel('X'), ylabel('Y');                                           % 显示坐标轴
saveas(gcf, ['DGA_result', '.bmp']);                                % 获取当前figure的窗口句柄,保存图片
toc
