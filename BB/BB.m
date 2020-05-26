%---------------------------------------- train sample  -----------------------------------------%
%  Funtion: solve paramenter(general), include phi~(alpha, beta);phi(j11)~(alpha_j11, beta_j11);phi(j10)~(alpha_j10, beta_j10)
%  Convention: i means sample, j means feature
clc; clear; tic; format long;                                  % clc-clear Command Window, clear-clear Workspace, tic��toc���Ժ�ʱ����, ��long��ʽ��ʾ

train_file = xlsread('BetaBernoulli-training-data.xls');       % ����train����     
train_samples = size(train_file, 1);                           % ��ȡtrain��samples
train_label = train_file(:, 7);                                % ȡ�������ǩ-��7��
train_data = train_file(:, 1:6);                               % ȡ��������ǩ-ǰ6��

Num_phi_j = zeros(6, 2, 2);         % ������������-��¼��Ŀ
Expect_phi_j = zeros(6, 2, 2);      % �������£���������[y=1�¸�feature������]��[y=0�¸�feature������]
Num_phi = zeros(1, 2);              % ͳ��sample���������ĸ�����[1,1]-����[1,2]-��
Expect_phi = zeros(1, 2);           % ���������[E(phi),1-E(phi)]

% ���׶��ٸ�alpha,beta?
% Num_phi(:,:,1),y=1�£���һ�г�ֵΪalpha_j11,�ڶ��г�ֵbeta_j11
% Num_phi(:,:,2),y=0�£���һ�г�ֵΪalpha_j10,�ڶ��г�ֵbeta_j10
alpha_j11 = 2; beta_j11 = 2; alpha_j10 = 2; beta_j10 = 2; alpha = 2; beta = 2;
Num_phi_j(:,1,1) = Num_phi_j(:,1,1) + alpha_j11; Num_phi_j(:,2,1) = Num_phi_j(:,2,1) + beta_j11;    % y=1
Num_phi_j(:,1,2) = Num_phi_j(:,1,2) + alpha_j10; Num_phi_j(:,2,2) = Num_phi_j(:,2,2) + beta_j10;    % y=0
Num_phi(1,1) = Num_phi(1,1) + alpha; Num_phi(1,2) = Num_phi(1,2) + beta;                            

% �������ַ�ʽ�ȼ�
% ��ʽ1 - �ӷ�ʵ��
% for i = 1:train_samples
%     feature = train_data(i, :)';                        % ͳ��feature(j)���������ĸ���   
%     if train_label(i, 1)                                % ͳ����������[phi(j11), phi(j01)]������
%         Num_phi_j(:, 1, 1) = Num_phi_j(:, 1, 1) + feature;  % phi(j11)
%         Num_phi_j(:, 2, 1) = Num_phi_j(:, 2, 1) + ~feature; % phi(j01)
%         Num_phi(1, 1) =  Num_phi(1, 1) + 1;         % sample����������
%     else                               % ͳ�Ƹ�������[phi(j10), phi(j00)]������
%         Num_phi_j(:, 1, 2) = Num_phi_j(:, 1, 2) + feature;  % phi(j10)
%         Num_phi_j(:, 2, 2) = Num_phi_j(:, 2, 2) + ~feature; % phi(j00)
%         Num_phi(1, 2) =  Num_phi(1, 2) + 1;         % sample����������
%     end
% end
% ��ʽ2 - �˷�ʵ��
Num_phi = Num_phi + [sum(train_label), sum(~train_label)];
Num_phi_j(:, :, 1) = Num_phi_j(:, :, 1) + [train_label'*train_data; train_label'*~train_data]';
Num_phi_j(:, :, 2) = Num_phi_j(:, :, 2) + [~train_label'*train_data; (~train_label')*(1-train_data)]';


% ����test��Ҫʹ����Щ����ֵ
Expect_phi(1, 1) = Num_phi(1,1)/(Num_phi(1,1)+Num_phi(1,2)); Expect_phi(1, 2) = 1-Expect_phi(1, 1);
for k = 1:2
   Expect_phi_j(:, 1, k) = Num_phi_j(:, 1, k)./(Num_phi_j(:, 1, k)+Num_phi_j(:, 2, k));  % E(�µ�phi_j11)
   Expect_phi_j(:, 2, k) = 1 - Expect_phi_j(:, 1, k);                                    % 1-E(�µ�phi_j11)
end

 clear i feature k;                  % �����Ч����


%---------------------------------------- test sample -----------------------------------------%
%  Funtion: test and evaluation
%  ���Ԥ�⣿����һ����������������Ϊ1��0�ĸ��ʣ�˭��Ԥ������˭��
%  test_result�����У���һ��������Ϊ1�ĸ��ʣ��ڶ���������Ϊ0�ĸ��ʣ�������Ԥ��ֵ(�Ƚ�ǰ���д�С)
%  test_result(:, 3)��test_answer�Ƚϣ��Ϳ��������������
test_data = xlsread('BetaBernoulli-testing-data.xls');              % ����test���ݣ�����sheet1
test_answer = xlsread('BetaBernoulli-testing-data.xls', 2);         % ����answer���ݣ�����sheet2
[test_samples, features] = size(test_data);                         % ��ȡtest��samples��features
test_result = zeros(test_samples, 3);                               % ������ԵĽ��
confusion_matrix = zeros(2, 2);                                     % ��������

for i = 1:test_samples                                              % �ȼ��� size(train_file, 1)
    feature = test_data(i, :)';                                     % test��feature
    p_y = [1 1];                                                    % ��һ��������Ϊ1�ĸ��ʣ��ڶ���������Ϊ0�ĸ���
    for j = 1:features
        if feature(j, 1)                                            % ����������w_i
            p_y(1, 1) = p_y(1, 1)*Expect_phi_j(j, 1, 1);
            p_y(1, 2) = p_y(1, 2)*Expect_phi_j(j, 1, 2);
        else
            p_y(1, 1) = p_y(1, 1)*Expect_phi_j(j, 2, 1);
            p_y(1, 2) = p_y(1, 2)*Expect_phi_j(j, 2, 2);
        end
    end
    test_result(i,1:2) = p_y.*Expect_phi;                           % һ����������������Ϊ1��0�ĸ���
    
    % �Ƚϴ�С��Ԥ�Ⲣ�����������
    if test_result(i, 1) > test_result(i, 2)                 % �б�Ϊ�����������Ϊ1
        test_result(i, 3) = 1;                                          % ����Ԥ����
        if test_answer(i, 1)                                            % Ԥ��Ϊ1��ʵ��Ϊ1
            confusion_matrix(1, 1) = confusion_matrix(1, 1) + 1;    % TP
        else                                                            % Ԥ��Ϊ1��ʵ��Ϊ0
            confusion_matrix(2, 1) = confusion_matrix(2, 1) + 1;    % FP
        end
    else                                                     % �б�Ϊ�����������Ϊ0
        test_result(i, 3) = 0;                                          % ����Ԥ����
        if test_answer(i, 1)                                            % Ԥ��Ϊ0��ʵ��Ϊ1
            confusion_matrix(1, 2) = confusion_matrix(1, 2) + 1;    % FN
        else                                                            % Ԥ��Ϊ0��ʵ��Ϊ0
            confusion_matrix(2, 2) = confusion_matrix(2, 2) + 1;    % TN
        end
    end
end

if exist('BB_testing_result.xlsx','file')                % �ļ����ھ�ɾ������ֹ�ϴεĽ��Ӱ��
    delete BB_testing_result.xlsx;
end
xlswrite('BB_testing_result.xlsx', test_result);         % д����
F1_score = 2*confusion_matrix(1, 1)/(length(test_data)+confusion_matrix(1, 1)-confusion_matrix(2, 2));
disp('confusion_matrix:'); disp(confusion_matrix);       % ��ʾ������-confusion_matrix
disp('F1_score:'); disp(F1_score);                       % ��ʾ������-F1_score
clear i feature k;                                       % �����Ч����
toc