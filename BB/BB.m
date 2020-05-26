%---------------------------------------- train sample  -----------------------------------------%
%  Funtion: solve paramenter(general), include phi~(alpha, beta);phi(j11)~(alpha_j11, beta_j11);phi(j10)~(alpha_j10, beta_j10)
%  Convention: i means sample, j means feature
clc; clear; tic; format long;                                  % clc-clear Command Window, clear-clear Workspace, tic与toc粗略耗时分析, 以long格式显示

train_file = xlsread('BetaBernoulli-training-data.xls');       % 读入train数据     
train_samples = size(train_file, 1);                           % 获取train集samples
train_label = train_file(:, 7);                                % 取出结果标签-第7列
train_data = train_file(:, 1:6);                               % 取出特征标签-前6列

Num_phi_j = zeros(6, 2, 2);         % 产生两个矩阵-记录数目
Expect_phi_j = zeros(6, 2, 2);      % 新数据下，两个矩阵[y=1下各feature的期望]和[y=0下各feature的期望]
Num_phi = zeros(1, 2);              % 统计sample正负样本的个数，[1,1]-正，[1,2]-负
Expect_phi = zeros(1, 2);           % 求出新期望[E(phi),1-E(phi)]

% 到底多少个alpha,beta?
% Num_phi(:,:,1),y=1下，第一列初值为alpha_j11,第二列初值beta_j11
% Num_phi(:,:,2),y=0下，第一列初值为alpha_j10,第二列初值beta_j10
alpha_j11 = 2; beta_j11 = 2; alpha_j10 = 2; beta_j10 = 2; alpha = 2; beta = 2;
Num_phi_j(:,1,1) = Num_phi_j(:,1,1) + alpha_j11; Num_phi_j(:,2,1) = Num_phi_j(:,2,1) + beta_j11;    % y=1
Num_phi_j(:,1,2) = Num_phi_j(:,1,2) + alpha_j10; Num_phi_j(:,2,2) = Num_phi_j(:,2,2) + beta_j10;    % y=0
Num_phi(1,1) = Num_phi(1,1) + alpha; Num_phi(1,2) = Num_phi(1,2) + beta;                            

% 下面两种方式等价
% 方式1 - 加法实现
% for i = 1:train_samples
%     feature = train_data(i, :)';                        % 统计feature(j)正负样本的个数   
%     if train_label(i, 1)                                % 统计正样本下[phi(j11), phi(j01)]及总数
%         Num_phi_j(:, 1, 1) = Num_phi_j(:, 1, 1) + feature;  % phi(j11)
%         Num_phi_j(:, 2, 1) = Num_phi_j(:, 2, 1) + ~feature; % phi(j01)
%         Num_phi(1, 1) =  Num_phi(1, 1) + 1;         % sample正样本个数
%     else                               % 统计负样本下[phi(j10), phi(j00)]及总数
%         Num_phi_j(:, 1, 2) = Num_phi_j(:, 1, 2) + feature;  % phi(j10)
%         Num_phi_j(:, 2, 2) = Num_phi_j(:, 2, 2) + ~feature; % phi(j00)
%         Num_phi(1, 2) =  Num_phi(1, 2) + 1;         % sample负样本个数
%     end
% end
% 方式2 - 乘法实现
Num_phi = Num_phi + [sum(train_label), sum(~train_label)];
Num_phi_j(:, :, 1) = Num_phi_j(:, :, 1) + [train_label'*train_data; train_label'*~train_data]';
Num_phi_j(:, :, 2) = Num_phi_j(:, :, 2) + [~train_label'*train_data; (~train_label')*(1-train_data)]';


% 下面test需要使用这些期望值
Expect_phi(1, 1) = Num_phi(1,1)/(Num_phi(1,1)+Num_phi(1,2)); Expect_phi(1, 2) = 1-Expect_phi(1, 1);
for k = 1:2
   Expect_phi_j(:, 1, k) = Num_phi_j(:, 1, k)./(Num_phi_j(:, 1, k)+Num_phi_j(:, 2, k));  % E(新的phi_j11)
   Expect_phi_j(:, 2, k) = 1 - Expect_phi_j(:, 1, k);                                    % 1-E(新的phi_j11)
end

 clear i feature k;                  % 清除无效变量


%---------------------------------------- test sample -----------------------------------------%
%  Funtion: test and evaluation
%  如何预测？对于一个新样本，计算它为1，0的概率，谁大预测结果跟谁。
%  test_result共三列，第一列新样本为1的概率，第二列新样本为0的概率，第三列预测值(比较前两列大小)
%  test_result(:, 3)和test_answer比较，就可以求出混淆矩阵
test_data = xlsread('BetaBernoulli-testing-data.xls');              % 读入test数据，即表sheet1
test_answer = xlsread('BetaBernoulli-testing-data.xls', 2);         % 读入answer数据，即表sheet2
[test_samples, features] = size(test_data);                         % 获取test集samples，features
test_result = zeros(test_samples, 3);                               % 保存测试的结果
confusion_matrix = zeros(2, 2);                                     % 混淆矩阵

for i = 1:test_samples                                              % 等价于 size(train_file, 1)
    feature = test_data(i, :)';                                     % test的feature
    p_y = [1 1];                                                    % 第一个新样本为1的概率，第二个新样本为0的概率
    for j = 1:features
        if feature(j, 1)                                            % 这里类似求w_i
            p_y(1, 1) = p_y(1, 1)*Expect_phi_j(j, 1, 1);
            p_y(1, 2) = p_y(1, 2)*Expect_phi_j(j, 1, 2);
        else
            p_y(1, 1) = p_y(1, 1)*Expect_phi_j(j, 2, 1);
            p_y(1, 2) = p_y(1, 2)*Expect_phi_j(j, 2, 2);
        end
    end
    test_result(i,1:2) = p_y.*Expect_phi;                           % 一个新样本，计算它为1，0的概率
    
    % 比较大小做预测并求出混淆矩阵
    if test_result(i, 1) > test_result(i, 2)                 % 判别为正样本，标记为1
        test_result(i, 3) = 1;                                          % 保存预测结果
        if test_answer(i, 1)                                            % 预测为1，实际为1
            confusion_matrix(1, 1) = confusion_matrix(1, 1) + 1;    % TP
        else                                                            % 预测为1，实际为0
            confusion_matrix(2, 1) = confusion_matrix(2, 1) + 1;    % FP
        end
    else                                                     % 判别为负样本，标记为0
        test_result(i, 3) = 0;                                          % 保存预测结果
        if test_answer(i, 1)                                            % 预测为0，实际为1
            confusion_matrix(1, 2) = confusion_matrix(1, 2) + 1;    % FN
        else                                                            % 预测为0，实际为0
            confusion_matrix(2, 2) = confusion_matrix(2, 2) + 1;    % TN
        end
    end
end

if exist('BB_testing_result.xlsx','file')                % 文件存在就删除，防止上次的结果影响
    delete BB_testing_result.xlsx;
end
xlswrite('BB_testing_result.xlsx', test_result);         % 写入结果
F1_score = 2*confusion_matrix(1, 1)/(length(test_data)+confusion_matrix(1, 1)-confusion_matrix(2, 2));
disp('confusion_matrix:'); disp(confusion_matrix);       % 显示处理结果-confusion_matrix
disp('F1_score:'); disp(F1_score);                       % 显示处理结果-F1_score
clear i feature k;                                       % 清除无效变量
toc