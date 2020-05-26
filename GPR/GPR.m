%---------------------------------------- train sample  -----------------------------------------%
%  Funtion: train(199个)和test(19801个)
%  GPR思路很简单，可能就是算不出来，如何解决耗时长？运行必须满足8G以上的物理内存，否则死机
%  为什么会需要这么多内存？Sigma_ij是2w*2w的浮点矩阵，常驻2G内存，后续还要做算术操作,保存到磁盘可能需要3G空间
%  为了清楚表示矩阵维度，不防记train个数为m1,test个数为m2; 希腊字母第一个字母大写表示大写；a表示train,b表示test
%  这里有两个sigma,都让它等于1
clc; clear; tic; format long;                       % clc-clear Command Window, clear-clear Workspace, tic与toc粗略耗时分析, 以long格式显示

file = xlsread('GRP_data.xls');                     % 读入train数据    
train_label = file(:, 6);                           % 取出结果标签-第6列
train_label = train_label(~isnan(train_label));     % 去除NAN数据，2016版本以上，也可以用rmmissing去除NAN数据
train_data = file(:, 1:5);                          % 取出特征标签-前5列   
clear file;                                         % 去除不需要的内存，否则物理内存>12G                  

% 计算出Sigma_ij，注意：切不可用for循环来实现（方式一）
% 1 使用for循环完成目的，单核至少要3个小时；方案二大概10秒
% 2 matlab的for效率远低于C/C++,在matlab中有很多函数和操作是基于内置for循环，同时也提供并行for结构-parfor，在一定程度上弥补这个缺陷
% 3 如果代码中实在需要大量for循环，建议使用mex调用C/C++的dll,对于vs2017，需要matlab2017及以上
% 4 仔细想来，for循环是串行结构，底层到汇编也只会分配单核计算，会发现cpu使用率一直是25%左右；pdist底层调用并行库，cpu使用可达>95%;我想这个结果
%   比调用c/c++都好，我觉得要充分利用编程工具和语言优势。
% 以上仅代表个人观点，还望多多指正
% 方式一
% sigma = 1;
% c = 1; d = 2;
%Sigma_ij = zeros(total_samples, total_samples);
% for i = 1:total_samples
%     feature_i = train_data(i, 1:features);
%     for j = i:total_samples
%         feature_j = train_data(j, 1:features);
%         Sigma_ij(i, j) = exp(-(pdist([feature_i;feature_j],'euclidean'))^2/(2*sigma^2));    % RBF-kernel计算              
%     end
% end
% 方式二
% pdist结果是1*(m(m-1)/2)的行向量，squareform将结果转成m*m的对称矩阵，主对角线全为0
Sigma_ij = squareform(exp(-0.5.*(pdist(train_data,'euclidean')).^2));     % pdist算两两间距离，可以指定其他距离； squareform将结果转成对称矩阵   
Sigma_ij = Sigma_ij + 2*eye(size(Sigma_ij));            % 主对角线全为0，这里乘了2

m1 = length(train_label);
% cells = mat2cell(Sigma_ij, [m1, m-m1], [m1,m-m1]);    % 矩阵分块，导致内存copy，效益并不高，释放某块内存-cells(i,j)=[],cells{i,j}=[]清零
Sigma_aa = Sigma_ij(1:m1, 1:m1);                        % 分割出各部分协方差,注意左上(m1*m1)是train
Sigma_ab = Sigma_ij(1:m1, m1+1:end);
Sigma_ba = Sigma_ij(m1+1:end, 1:m1);
Sigma_bb = Sigma_ij(m1+1:end, m1+1:end);
clear Sigma_ij;                                         % 清除Sigma_ij

temp = Sigma_ba*inv(Sigma_aa);
result_mu = temp*train_label;                           % 计算在train条件下test的均值-19801*1
clear Sigma_ba Sigma_aa train_label train_data;         % 其实这里可以添加一些空操作，因为可能导致内存尖峰（短时间需要很大内存，clear来不急及时释放），若是8G内存就死机了
result_sigma = Sigma_bb-temp*Sigma_ab;                  % 计算在train条件下test的协方差-19801*19801
clear m1 Sigma_ab Sigma_bb temp;
toc
