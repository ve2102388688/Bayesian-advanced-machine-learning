%---------------------------------------- train sample  -----------------------------------------%
%  Funtion: solve posterior distribution(mu, Lambda), 
%  Convention: i means sample, j means feature
clc; clear; tic; format long;                                  % clc-clear Command Window, clear-clear Workspace, tic��toc���Ժ�ʱ����, ��long��ʽ��ʾ

train_file = xlsread('GG_training_data.xls');                  % ����train����     

train_label = train_file(:, 6);                                % ȡ�������ǩ-��7��
train_data = train_file(:, 1:5);                               % ȡ��������ǩ-ǰ6��
features = size(train_data, 2);                                % ��ȡtrain��samples

% ��������
a = 1;
b = 1;                                                         % ������y�ֲ���lambda_y���Ƶ���b
Lambda = b*train_data'*train_data + a*eye(features);           % ������� Lambda
mu = b*inv(Lambda)*train_data'*train_label;                    % ������� mu


%---------------------------------------- test sample -----------------------------------------%
%  Funtion: test and evaluation
%  ���Ԥ�⣿����һ����������������ֲ�y(mu_y, 1/lambda_y),Ԥ��ֵΪmu_y
%  ���������Ԥ������𰸵����ϵ����ͨ������rho=0.996,˵��Ԥ���ֵ�ʹ��Ǹ߶Ƚӽ��ģ��Ҳ���a.b��ʼֵӰ��
test_data = xlsread('GG_testing_data.xls');              % ����test���ݣ�����sheet1
test_answer = xlsread('GG_testing_data.xls', 2);         % ����answer���ݣ�����sheet2
test_samples = length(test_data);                        % ��ȡtest��samples
test_result = zeros(test_samples, 3);                    % ������ԵĽ��

for i = 1:test_samples                                   % �ȼ��� size(train_file, 1)
    feature = test_data(i, :)';                          % test��feature
    % ����Ԥ��ֲ�y
    Lambda1_inv = inv(b*feature*feature'+ Lambda);
    lambda_y = b-b^2*feature'*Lambda1_inv*feature;
    mu_y = b*mu'*Lambda*Lambda1_inv*feature/lambda_y;
    % ������
    test_result(i, 1:2) = [mu_y, 1/lambda_y];                   % ��¼�ֲ�y�ľ�ֵ�ͷ���
    test_result(i, 3) =  test_result(i, 1)-test_answer(i, 1);   % ���e
end

%------------------------------- evaluation -----------------------%
corr = corrcoef(test_result(:, 1), test_answer(:, 1));   % Ԥ������𰸵����ϵ��-��֤Ԥ��ֵ�Ƿ�ʹ��ǽӽ��ģ�
disp('Ԥ������𰸵����ϵ����'); disp(corr(1,2))

if exist('GG_testing_result.xlsx','file')                % �ļ����ھ�ɾ������ֹ�ϴεĽ��Ӱ��
    delete GG_testing_result.xlsx;
end
xlswrite('GG_testing_result.xlsx', test_result);         % д����

clear i feature;                                         % �����Ч����
toc

