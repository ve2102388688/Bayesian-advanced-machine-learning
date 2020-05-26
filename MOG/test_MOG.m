clc; clear; tic; format long;                % clc-clear Command Window, clear-clear Workspace, tic与toc粗略耗时分析, 以long格式显示
train_data = xlsread('MoG_data.xls');        % 读入train数据     
if exist('result.xls','file')                % 文件存在就删除，防止上次的结果影响
    delete result.xls;
end
init_phi = 0.1:0.1:0.9;                      % phi初始值从0.1到0.9
for i = 1:length(init_phi)
     MOG(train_data, init_phi(i), (i-1)*2);  % 建议 help MOG
end

plot_res = xlsread('result.xls');            % 读入result数据    
figure('NumberTitle', 'off', 'Name', 'MOB'); hold on; grid on; title('\phi从0.1到0.9');
xlabel('迭代次数k'), ylabel('\phi初始值'); axis([1 40 0 1])
set(gca,'ytick',0:0.05:1);                   % 设置y轴显示精度
for i = 1:2:length(init_phi)*2
    plot(plot_res(:, i), plot_res(:, i+1));  % 同一坐标，画出所有曲线
end
date_npw = datestr(now, '_HH_MM_SS');        % 以时间后缀存储，避免重名
saveas(gcf, ['MOB',date_npw, '.bmp']);       % 获取当前figure的窗口句柄,保存图片
save(['result', date_npw], 'plot_res');      % 存储对应的result****.m
toc
