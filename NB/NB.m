%---------------------------------------- train sample -----------------------------------------%
%  Funtion: solve paramenter(general), include phi,phi(j11),phi(j10).
%  Convention: i means sample, j means feature
%  注意：注释中,与;是存储结构的具体解释，如[phi(j11), phi(j01)]表示第一列是phi(j11)，第二列phi(j01)
clc; clear; tic; format long;                       % clc-clear Command Window, clear-clear Workspace, tic与toc粗略耗时分析, 以long格式显示

train_file = xlsread('NB_training_data.xls');       % 读入train数据  
train_samples = size(train_file, 1);                % 获取train集samples
train_label = train_file(:, 7);                     % 取出结果标签-第7列
train_data = train_file(:, 1:6);                    % 取出特征标签feature-前6列，还记得GDA被简化成X,Y吗？

Num_phi = zeros(6, 2, 2);         % 产生两个矩阵-记录数目
Num_totol = zeros(2, 1);          % 统计sample正负样本的个数，[1,1]-正，[2,1]-负
Prob_train = zeros(6, 2, 2);      % 产生两个矩阵-[phi(j11), phi(j01)]和[phi(j10), phi(j00)]，记录百分比

% 下面两种方式等价
% 方式1 - 加法实现
% 举个例子，若j=1, phi(111)+phi(101)+phi(110)+phi(100) = 样本总数，j取其他值也是这样，就可以理解下面的feature与~feature
% for i = 1:train_samples
%     feature = train_data(i, :)';                        % 统计feature(j)正负样本的个数   
%     if train_label(i, 1)                                % 统计正样本下[phi(j11), phi(j01)]及总数
%         Num_phi(:, 1, 1) = Num_phi(:, 1, 1) + feature;  % phi(j11), 由于feature是二值，非黑即白，不是黑的，就是白的（下面取反）
%         Num_phi(:, 2, 1) = Num_phi(:, 2, 1) + ~feature; % phi(j01)
%         Num_totol(1, 1) =  Num_totol(1, 1) + 1;         % sample正样本个数+1
%     else                       % 统计负样本下[phi(j10), phi(j00)]及总数
%         Num_phi(:, 1, 2) = Num_phi(:, 1, 2) + feature;  % phi(j10)
%         Num_phi(:, 2, 2) = Num_phi(:, 2, 2) + ~feature; % phi(j00)
%         Num_totol(2, 1) =  Num_totol(2, 1) + 1;         % sample负样本个数+1
%     end
% end
% 方式2 - 乘法实现
Num_totol(:) = [sum(train_label); sum(~train_label)];
Num_phi(:, :, 1) = [train_label'*train_data; train_label'*~train_data]';
Num_phi(:, :, 2) = [~train_label'*train_data; (~train_label')*(1-train_data)]';

for k = 1:2
   Prob_train(:, :, k) = Num_phi(:, :, k)/Num_totol(k, 1);  % 将数目转换成百分比
end

clear i feature k;                      % 清除无效变量


%---------------------------------------- test sample -----------------------------------------%
%  Funtion: test and evaluation
%  如何预测？对于一个新样本，计算它为1，0的概率，谁大预测结果跟谁。
%  test_result共三列，第一列新样本为1的概率，第二列新样本为0的概率，第三列预测值(比较前两列大小)
%  test_result(:, 3)和test_answer比较，就可以求出混淆矩阵
test_data = xlsread('NB_testing_data.xls');                 % 读入test数据，即表sheet1
test_answer = xlsread('NB_testing_data.xls', 2);            % 读入answer数据，即表sheet2
[test_samples, features] = size(test_data);                 % 获取test集samples，features
test_result = zeros(test_samples, 3);                       % 保存测试的结果
confusion_matrix = zeros(2, 2);                             % 混淆矩阵

for i = 1:test_samples                                      % 等价于 size(train_file, 1)
    feature = test_data(i, :)';                             % test的feature
    for k = 1:2     % k=1,计算新样本为1的概率， k=0,计算新样本为0的概率
        test_result(i, k) = prod(feature.*Prob_train(:, 1, k) + ~feature.*Prob_train(:, 2, k));      % 一个新样本，计算它为1，0的概率
    end
    
    % 比较大小做预测并求出混淆矩阵
    if test_result(i, 1) > test_result(i, 2)               % 判别为正样本，标记为1
        test_result(i, 3) = 1;                             % 保存预测结果
        if test_answer(i, 1)                               % 预测为1，实际为1
            confusion_matrix(1, 1) = confusion_matrix(1, 1) + 1;    % TP
        else                                               % 预测为1，实际为0
            confusion_matrix(2, 1) = confusion_matrix(2, 1) + 1;    % FP
        end
    else                                                   % 判别为负样本，标记为0
        test_result(i, 3) = 0;                             % 保存预测结果
        if test_answer(i, 1)                               % 预测为0，实际为1
            confusion_matrix(1, 2) = confusion_matrix(1, 2) + 1;    % FN
        else                                               % 预测为0，实际为0
            confusion_matrix(2, 2) = confusion_matrix(2, 2) + 1;    % TN
        end
    end
end

if exist('NB_testing_result.xlsx','file')                % 文件存在就删除，防止上次的结果影响
    delete NB_testing_result.xlsx;
end
xlswrite('NB_testing_result.xlsx', test_result);         % 写入结果
F1_score = 2*confusion_matrix(1, 1)/(length(test_data)+confusion_matrix(1, 1)-confusion_matrix(2, 2));
disp('confusion_matrix:'); disp(confusion_matrix);       % 显示处理结果-confusion_matrix
disp('F1_score:'); disp(F1_score);                       % 显示处理结果-F1_score
clear i feature k;                                       % 清除无效变量
toc