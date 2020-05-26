%---------------------------------------- train sample  -----------------------------------------%
%  Funtion: train(199��)��test(19801��)
%  GPR˼·�ܼ򵥣����ܾ����㲻��������ν����ʱ�������б�������8G���ϵ������ڴ棬��������
%  Ϊʲô����Ҫ��ô���ڴ棿Sigma_ij��2w*2w�ĸ�����󣬳�פ2G�ڴ棬������Ҫ����������,���浽���̿�����Ҫ3G�ռ�
%  Ϊ�������ʾ����ά�ȣ�������train����Ϊm1,test����Ϊm2; ϣ����ĸ��һ����ĸ��д��ʾ��д��a��ʾtrain,b��ʾtest
%  ����������sigma,����������1
clc; clear; tic; format long;                       % clc-clear Command Window, clear-clear Workspace, tic��toc���Ժ�ʱ����, ��long��ʽ��ʾ

file = xlsread('GRP_data.xls');                     % ����train����    
train_label = file(:, 6);                           % ȡ�������ǩ-��6��
train_label = train_label(~isnan(train_label));     % ȥ��NAN���ݣ�2016�汾���ϣ�Ҳ������rmmissingȥ��NAN����
train_data = file(:, 1:5);                          % ȡ��������ǩ-ǰ5��   
clear file;                                         % ȥ������Ҫ���ڴ棬���������ڴ�>12G                  

% �����Sigma_ij��ע�⣺�в�����forѭ����ʵ�֣���ʽһ��
% 1 ʹ��forѭ�����Ŀ�ģ���������Ҫ3��Сʱ�����������10��
% 2 matlab��forЧ��Զ����C/C++,��matlab���кܶຯ���Ͳ����ǻ�������forѭ����ͬʱҲ�ṩ����for�ṹ-parfor����һ���̶����ֲ����ȱ��
% 3 ���������ʵ����Ҫ����forѭ��������ʹ��mex����C/C++��dll,����vs2017����Ҫmatlab2017������
% 4 ��ϸ������forѭ���Ǵ��нṹ���ײ㵽���Ҳֻ����䵥�˼��㣬�ᷢ��cpuʹ����һֱ��25%���ң�pdist�ײ���ò��п⣬cpuʹ�ÿɴ�>95%;����������
%   �ȵ���c/c++���ã��Ҿ���Ҫ������ñ�̹��ߺ��������ơ�
% ���Ͻ�������˹۵㣬�������ָ��
% ��ʽһ
% sigma = 1;
% c = 1; d = 2;
%Sigma_ij = zeros(total_samples, total_samples);
% for i = 1:total_samples
%     feature_i = train_data(i, 1:features);
%     for j = i:total_samples
%         feature_j = train_data(j, 1:features);
%         Sigma_ij(i, j) = exp(-(pdist([feature_i;feature_j],'euclidean'))^2/(2*sigma^2));    % RBF-kernel����              
%     end
% end
% ��ʽ��
% pdist�����1*(m(m-1)/2)����������squareform�����ת��m*m�ĶԳƾ������Խ���ȫΪ0
Sigma_ij = squareform(exp(-0.5.*(pdist(train_data,'euclidean')).^2));     % pdist����������룬����ָ���������룻 squareform�����ת�ɶԳƾ���   
Sigma_ij = Sigma_ij + 2*eye(size(Sigma_ij));            % ���Խ���ȫΪ0���������2

m1 = length(train_label);
% cells = mat2cell(Sigma_ij, [m1, m-m1], [m1,m-m1]);    % ����ֿ飬�����ڴ�copy��Ч�沢���ߣ��ͷ�ĳ���ڴ�-cells(i,j)=[],cells{i,j}=[]����
Sigma_aa = Sigma_ij(1:m1, 1:m1);                        % �ָ��������Э����,ע������(m1*m1)��train
Sigma_ab = Sigma_ij(1:m1, m1+1:end);
Sigma_ba = Sigma_ij(m1+1:end, 1:m1);
Sigma_bb = Sigma_ij(m1+1:end, m1+1:end);
clear Sigma_ij;                                         % ���Sigma_ij

temp = Sigma_ba*inv(Sigma_aa);
result_mu = temp*train_label;                           % ������train������test�ľ�ֵ-19801*1
clear Sigma_ba Sigma_aa train_label train_data;         % ��ʵ����������һЩ�ղ�������Ϊ���ܵ����ڴ��壨��ʱ����Ҫ�ܴ��ڴ棬clear��������ʱ�ͷţ�������8G�ڴ��������
result_sigma = Sigma_bb-temp*Sigma_ab;                  % ������train������test��Э����-19801*19801
clear m1 Sigma_ab Sigma_bb temp;
toc
