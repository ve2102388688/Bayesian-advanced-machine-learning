%---------------------------------------- train sample -----------------------------------------%
%  Funtion: solve paramenter(general), include phi,phi(j11),phi(j10).
%  Convention: i means sample, j means feature
%  ע�⣺ע����,��;�Ǵ洢�ṹ�ľ�����ͣ���[phi(j11), phi(j01)]��ʾ��һ����phi(j11)���ڶ���phi(j01)
clc; clear; tic; format long;                       % clc-clear Command Window, clear-clear Workspace, tic��toc���Ժ�ʱ����, ��long��ʽ��ʾ

train_file = xlsread('NB_training_data.xls');       % ����train����  
train_samples = size(train_file, 1);                % ��ȡtrain��samples
train_label = train_file(:, 7);                     % ȡ�������ǩ-��7��
train_data = train_file(:, 1:6);                    % ȡ��������ǩfeature-ǰ6�У����ǵ�GDA���򻯳�X,Y��

Num_phi = zeros(6, 2, 2);         % ������������-��¼��Ŀ
Num_totol = zeros(2, 1);          % ͳ��sample���������ĸ�����[1,1]-����[2,1]-��
Prob_train = zeros(6, 2, 2);      % ������������-[phi(j11), phi(j01)]��[phi(j10), phi(j00)]����¼�ٷֱ�

% �������ַ�ʽ�ȼ�
% ��ʽ1 - �ӷ�ʵ��
% �ٸ����ӣ���j=1, phi(111)+phi(101)+phi(110)+phi(100) = ����������jȡ����ֵҲ���������Ϳ�����������feature��~feature
% for i = 1:train_samples
%     feature = train_data(i, :)';                        % ͳ��feature(j)���������ĸ���   
%     if train_label(i, 1)                                % ͳ����������[phi(j11), phi(j01)]������
%         Num_phi(:, 1, 1) = Num_phi(:, 1, 1) + feature;  % phi(j11), ����feature�Ƕ�ֵ���Ǻڼ��ף����Ǻڵģ����ǰ׵ģ�����ȡ����
%         Num_phi(:, 2, 1) = Num_phi(:, 2, 1) + ~feature; % phi(j01)
%         Num_totol(1, 1) =  Num_totol(1, 1) + 1;         % sample����������+1
%     else                       % ͳ�Ƹ�������[phi(j10), phi(j00)]������
%         Num_phi(:, 1, 2) = Num_phi(:, 1, 2) + feature;  % phi(j10)
%         Num_phi(:, 2, 2) = Num_phi(:, 2, 2) + ~feature; % phi(j00)
%         Num_totol(2, 1) =  Num_totol(2, 1) + 1;         % sample����������+1
%     end
% end
% ��ʽ2 - �˷�ʵ��
Num_totol(:) = [sum(train_label); sum(~train_label)];
Num_phi(:, :, 1) = [train_label'*train_data; train_label'*~train_data]';
Num_phi(:, :, 2) = [~train_label'*train_data; (~train_label')*(1-train_data)]';

for k = 1:2
   Prob_train(:, :, k) = Num_phi(:, :, k)/Num_totol(k, 1);  % ����Ŀת���ɰٷֱ�
end

clear i feature k;                      % �����Ч����


%---------------------------------------- test sample -----------------------------------------%
%  Funtion: test and evaluation
%  ���Ԥ�⣿����һ����������������Ϊ1��0�ĸ��ʣ�˭��Ԥ������˭��
%  test_result�����У���һ��������Ϊ1�ĸ��ʣ��ڶ���������Ϊ0�ĸ��ʣ�������Ԥ��ֵ(�Ƚ�ǰ���д�С)
%  test_result(:, 3)��test_answer�Ƚϣ��Ϳ��������������
test_data = xlsread('NB_testing_data.xls');                 % ����test���ݣ�����sheet1
test_answer = xlsread('NB_testing_data.xls', 2);            % ����answer���ݣ�����sheet2
[test_samples, features] = size(test_data);                 % ��ȡtest��samples��features
test_result = zeros(test_samples, 3);                       % ������ԵĽ��
confusion_matrix = zeros(2, 2);                             % ��������

for i = 1:test_samples                                      % �ȼ��� size(train_file, 1)
    feature = test_data(i, :)';                             % test��feature
    for k = 1:2     % k=1,����������Ϊ1�ĸ��ʣ� k=0,����������Ϊ0�ĸ���
        test_result(i, k) = prod(feature.*Prob_train(:, 1, k) + ~feature.*Prob_train(:, 2, k));      % һ����������������Ϊ1��0�ĸ���
    end
    
    % �Ƚϴ�С��Ԥ�Ⲣ�����������
    if test_result(i, 1) > test_result(i, 2)               % �б�Ϊ�����������Ϊ1
        test_result(i, 3) = 1;                             % ����Ԥ����
        if test_answer(i, 1)                               % Ԥ��Ϊ1��ʵ��Ϊ1
            confusion_matrix(1, 1) = confusion_matrix(1, 1) + 1;    % TP
        else                                               % Ԥ��Ϊ1��ʵ��Ϊ0
            confusion_matrix(2, 1) = confusion_matrix(2, 1) + 1;    % FP
        end
    else                                                   % �б�Ϊ�����������Ϊ0
        test_result(i, 3) = 0;                             % ����Ԥ����
        if test_answer(i, 1)                               % Ԥ��Ϊ0��ʵ��Ϊ1
            confusion_matrix(1, 2) = confusion_matrix(1, 2) + 1;    % FN
        else                                               % Ԥ��Ϊ0��ʵ��Ϊ0
            confusion_matrix(2, 2) = confusion_matrix(2, 2) + 1;    % TN
        end
    end
end

if exist('NB_testing_result.xlsx','file')                % �ļ����ھ�ɾ������ֹ�ϴεĽ��Ӱ��
    delete NB_testing_result.xlsx;
end
xlswrite('NB_testing_result.xlsx', test_result);         % д����
F1_score = 2*confusion_matrix(1, 1)/(length(test_data)+confusion_matrix(1, 1)-confusion_matrix(2, 2));
disp('confusion_matrix:'); disp(confusion_matrix);       % ��ʾ������-confusion_matrix
disp('F1_score:'); disp(F1_score);                       % ��ʾ������-F1_score
clear i feature k;                                       % �����Ч����
toc