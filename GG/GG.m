%---------------------------------------- train sample  -----------------------------------------%
%  Funtion: solve posterior distribution(mu, Lambda), 
%  Convention: i means sample, j means feature
clc; clear; tic; format long;                                  % clc-clear Command Window, clear-clear Workspace, tic与toc粗略耗时分析, 以long格式显示

train_file = xlsread('GG_training_data.xls');                  % 读入train数据     

train_label = train_file(:, 6);                                % 取出结果标签-第7列
train_data = train_file(:, 1:5);                               % 取出特征标签-前6列
features = size(train_data, 2);                                % 获取train集samples

% 求后验概率
a = 1;
b = 1;                                                         % 最后求得y分布的lambda_y近似等于b
Lambda = b*train_data'*train_data + a*eye(features);           % 计算后验 Lambda
mu = b*inv(Lambda)*train_data'*train_label;                    % 计算后验 mu


%---------------------------------------- test sample -----------------------------------------%
%  Funtion: test and evaluation
%  如何预测？对于一个新样本，计算其分布y(mu_y, 1/lambda_y),预测值为mu_y
%  如何评估？预测结果与答案的相关系数，通过分析rho=0.996,说明预测的值和答案是高度接近的，且不随a.b初始值影响
test_data = xlsread('GG_testing_data.xls');              % 读入test数据，即表sheet1
test_answer = xlsread('GG_testing_data.xls', 2);         % 读入answer数据，即表sheet2
test_samples = length(test_data);                        % 获取test集samples
test_result = zeros(test_samples, 3);                    % 保存测试的结果

for i = 1:test_samples                                   % 等价于 size(train_file, 1)
    feature = test_data(i, :)';                          % test的feature
    % 计算预测分布y
    Lambda1_inv = inv(b*feature*feature'+ Lambda);
    lambda_y = b-b^2*feature'*Lambda1_inv*feature;
    mu_y = b*mu'*Lambda*Lambda1_inv*feature/lambda_y;
    % 保存结果
    test_result(i, 1:2) = [mu_y, 1/lambda_y];                   % 记录分布y的均值和方差
    test_result(i, 3) =  test_result(i, 1)-test_answer(i, 1);   % 误差e
end

%------------------------------- evaluation -----------------------%
corr = corrcoef(test_result(:, 1), test_answer(:, 1));   % 预测结果与答案的相关系数-验证预测值是否和答案是接近的？
disp('预测结果与答案的相关系数：'); disp(corr(1,2))

if exist('GG_testing_result.xlsx','file')                % 文件存在就删除，防止上次的结果影响
    delete GG_testing_result.xlsx;
end
xlswrite('GG_testing_result.xlsx', test_result);         % 写入结果

clear i feature;                                         % 清除无效变量
toc

