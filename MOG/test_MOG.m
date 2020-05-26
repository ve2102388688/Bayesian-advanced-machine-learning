clc; clear; tic; format long;                % clc-clear Command Window, clear-clear Workspace, tic��toc���Ժ�ʱ����, ��long��ʽ��ʾ
train_data = xlsread('MoG_data.xls');        % ����train����     
if exist('result.xls','file')                % �ļ����ھ�ɾ������ֹ�ϴεĽ��Ӱ��
    delete result.xls;
end
init_phi = 0.1:0.1:0.9;                      % phi��ʼֵ��0.1��0.9
for i = 1:length(init_phi)
     MOG(train_data, init_phi(i), (i-1)*2);  % ���� help MOG
end

plot_res = xlsread('result.xls');            % ����result����    
figure('NumberTitle', 'off', 'Name', 'MOB'); hold on; grid on; title('\phi��0.1��0.9');
xlabel('��������k'), ylabel('\phi��ʼֵ'); axis([1 40 0 1])
set(gca,'ytick',0:0.05:1);                   % ����y����ʾ����
for i = 1:2:length(init_phi)*2
    plot(plot_res(:, i), plot_res(:, i+1));  % ͬһ���꣬������������
end
date_npw = datestr(now, '_HH_MM_SS');        % ��ʱ���׺�洢����������
saveas(gcf, ['MOB',date_npw, '.bmp']);       % ��ȡ��ǰfigure�Ĵ��ھ��,����ͼƬ
save(['result', date_npw], 'plot_res');      % �洢��Ӧ��result****.m
toc
