%---------------------------------------- train sample -----------------------------------------%
%  Funtion: solve paramenter(general), include phi,[mu0-2*1,sigma0-2*2],[mu1-2*1,sigma1-2*2]
%  ����ᶨ�弸���������洢��Щֵ�����������������ԣ��ͻᶨ����ص����ݽṹ����ʾ������matlab���Ƿ����
%  Convention: i means sample, j means feature;�����ά��̬�ֲ��������ǵ�һ������ΪX���ڶ�������ΪY
clc; clear; tic;                                 % clc-clear Command Window, clear-clear Workspace, tic��toc���Ժ�ʱ����            

train_file = xlsread('GDA_training_data.xls');   % ����train����    

% ɸѡ����-Ҳ������find,���ڱ�ǩ�Ƕ�ֵ���ݣ�����ʡ��find
train_data_y1 = train_file(train_file(:, 3) == 1, :);   % train_data_y1�Ǳ�ǩΪ1���������ݣ�������ǩ
train_data_y0 = train_file(train_file(:, 3) == 0, :);   % train_data_y1�Ǳ�ǩΪ0���������ݣ�������ǩ

mu = zeros(3, 1, 2);                                         % ������������-[mu1(X);mu1(Y)]��[mu0(X);mu0(Y)],���һ���Ǳ�ǩ-�ɺ���
sigma = zeros(2, 2, 2);                                      % ������������-�洢Э����
Num_totol = [length(train_data_y1); length(train_data_y0)];  % sample���������ĸ�����[1,1]-����[2,1]-��

% ������\theta
phi = Num_totol(1, 1)/sum(Num_totol);                               % phi
mu(:, 1, 1) = mean(train_data_y1)';                                 % y=1�µ�[mu1(X);mu1(Y)]
mu(:, 1, 2) = mean(train_data_y0)';                                 % y=0�µ�[mu0(X);mu0(Y)]
sigma(:, :, 1) =  cov(train_data_y1(:, 1), train_data_y1(:, 2));    % y=1�µ�sigma1
sigma(:, :, 2) =  cov(train_data_y0(:, 1), train_data_y0(:, 2));    % y=0�µ�sigma0

scatter(train_data_y1(:, 1), train_data_y1(:, 2), '.'); hold on;    % ����ɢ��ͼ y=1
scatter(train_data_y0(:, 1), train_data_y0(:, 2), '.', 'r');        % y=0
xlabel('X'), ylabel('Y');                                           % ��ʾ������
saveas(gcf, ['DGA_result', '.bmp']);                                % ��ȡ��ǰfigure�Ĵ��ھ��,����ͼƬ
toc
