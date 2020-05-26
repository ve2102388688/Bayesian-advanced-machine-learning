function  MOB(train_data, init_phi, times)
%MOB Summary of this function goes here
%   train_data: 训练集
%   init_phi:   phi的初始值，最终于收敛到0.6954或0.3047
%   times:      希望train几次data,每次结果[k,phi]写入result不同位置
%   注意: feature维度为6，即j=1:6

samples = size(train_data,1); features = size(train_data,2); % 20000个样本，每个样本6维

% phi [phi(j11),phi(j01)] [phi(j10),phi(j00)] 这里j=1:6
% phi(1:1)->phi, phi(1:2)->(1-phi)
% phi_j(:,:,1)->z=1时,phi(j@1)，6*2维，第一列@=1， 第二列@=0（相当于第一列取反），'@'表示feature的取值，这里是0或1
% phi_j(:,:,2)->z=0时,phi(j@1)，6*2维，第一列@=1， 第二列@=0（相当于第一列取反），'@'表示feature的取值，这里是0或1
phi = zeros(1, 2);
phi_j = zeros(6, 2, 2);                         % 第一个矩阵[phi(j11),phi(j01)], 第二个矩阵[phi(j10),phi(j00)]
% 初始化，通过实验，可以发现phi最终值趋近0.6954或0.3047
phi(1, 1) = init_phi; phi(1, 2) = 1 - phi(1, 1);                                          % 初始值0.1到0.9
phi_j(:, 1, 1) = unifrnd(0.45, 0.55, features, 1); phi_j(:, 2, 1) = 1-phi_j(:, 1, 1);     % 这里phi(j11)均匀分布初始化，其他分布初始化也可以        
phi_j(:, 1, 2) = unifrnd(0.45, 0.55, features, 1); phi_j(:, 2, 2) = 1-phi_j(:, 1, 2);     % 这里phi(j10)均匀分布初始化，其他分布初始化也可以        

result = zeros(100, 2);     % 输出结果，[循环次数, 对应的\phi]
last_phi = 1;               % matlab 没有do-while循环，设置为1，好满足第一次判断条件
k = 1;                      % 记录迭代次数
w_i = zeros(samples, 1);  	% 每个样本都有一个w(i)，共20000个
while (abs(last_phi-phi(1, 1)) > 10^(-6))    % 这里要用绝对值，曲线是上升还是下降，不可预知
    disp(last_phi-phi(1, 1));                % 观察误差
    last_phi = phi(1, 1);
    
    result(k,:)  = [k, phi(1, 1)];           % 保存结果，观察每次的phi取值变化
    k = k+1;                                 % 迭代次数
    
    % 求取 w(i) = p_x_z1/(p_x_z1+p_x_z0)
    % 例如第一个feature=[1 0 1 1 1 0],则p_x_z1=phi(111)*(1-phi(211))*phi(311)*phi(411)*phi(511)*(1-phi(611))*phi
    % p_x_z0=phi(110)*(1-phi(210))*phi(310)*phi(410)*phi(510)*(1-phi(610))*(1-phi)
    % 更新w(i)，共20000个
    for i = 1:samples
        feature = train_data(i, :)';                % 每个样本的feature
        p_x_z1 = 1;
        p_x_z0 = 1;
        for j = 1:features
            if feature(j, 1)
                p_x_z1 = p_x_z1*phi_j(j, 1, 1);
                p_x_z0 = p_x_z0*phi_j(j, 1, 2);
            else
                p_x_z1 = p_x_z1*phi_j(j, 2, 1);
                p_x_z0 = p_x_z0*phi_j(j, 2, 2);
            end
        end
        p_x_z1 =  p_x_z1*phi(1,1);
        p_x_z0 =  p_x_z0*phi(1,2);
        w_i(i, 1) = p_x_z1 / (p_x_z0 + p_x_z1);
    end
    
    phi(1, 1) = sum(w_i)/samples; phi(1, 2) = 1 - phi(1, 1);                                                            % 更新phi及1-phi
    for j = 1:features
        phi_j(j, 1, 1) = sum(w_i(:, 1).*train_data(:, j)) / sum(w_i); phi_j(j, 2, 1) = 1-phi_j(j, 1, 1);                % 更新phi_j第一个矩阵
        phi_j(j, 1, 2) = sum((1-w_i(:, 1)).*train_data(:, j)) / (samples-sum(w_i)); phi_j(j, 2, 2) = 1-phi_j(j, 1, 2);  % 更新phi_j第二个矩阵
    end
end

clear i j;                                      % 清除无效循环变量
num2str(result(:, 2), '%.30g');                 % 提高保存精度
position = [char(times+65), num2str(1)];        % 获取excl的存放位置，如'A1','C1','E1',..
xlswrite('result.xls',  result, 1, position);   % 按'A1','C1','E1',..存放数据

end

