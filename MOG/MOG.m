function  MOG(train_data, init_phi, times)
%MOG Summary of this function goes here
%   train_data: 训练集
%   init_phi:   示例，研究了phi初值从0.1到0.9
%   times:      希望train几次data,每次结果[k,phi]写入result不同位置，运行完，建议打开result.xls
%   注意: 通常情况下，应叫做feature(如MOB有6列)，这里不妨记作X,Y(两列)，不用feature了

samples = size(train_data,1); %features = size(train_data,2);    % 20000个样本，每个样本2维，记作X_Y

% phi(1:1)->phi, phi(1:2)->(1-phi)
% mu(:,1,1)->mu(1), mu(:,1,2)->mu(0)
% sigma(:,:,1)->sigma(1)-2*2, sigma(:,:,2)->sigma(0)-2*2
phi = zeros(1, 2);                           % 预先分配，效率高
mu = zeros(2, 1, 2);                         % 第一个矩阵[mu(1)_X; mu(1)_Y], 第二个矩阵[mu(0)_X; mu(0)_Y]
sigma1 =  zeros(2, 2);                       % sigma1,Z=1
sigma0 =  zeros(2, 2);                       % sigma0,Z=0

% 初始化，这里的sigma1/sigma0初始化需要考虑其特点，保证rho属于[0,1],不要随机初始化，否则w_i会出现复数
% 通过实验，phi会收敛0.3102或0.6987，MOG初始化值比较多，下面繁琐的独立出每个变量，你可以任意设置，观察输出
% 这里让rho从0.1到0.9，其他参数都取值常数，你可以改变其他初值，如/mu1/mu0/sigma_X1/sigma_Y1/sigma_X0/sigma_Y0
phi(1, 1) = init_phi; phi(1, 2) = 1 - phi(1, 1);    % 随机取值，第二列是第一列概率意义下取反
mu(:, 1, 1) = [2; 6];                               % mu1常值，收敛结果是[5.0; 4.99]
mu(:, 1, 2) = [1; 3];                               % mu0常值，收敛结果是[0.002; 0.025]
rand_temp = [1 5; 2 6];                             % [sigma_X1 sigma_Y1; sigma_X0 sigma_Y0]
init_pho = 0.9;                                     % rho属于[0,1]
rho = [init_pho, init_pho];                         % 为了简化，可以使得z=1/z=0下rho相同
sigma1(:, :) = [rand_temp(1,1) rho(1,1)*sqrt(rand_temp(1,1)*rand_temp(1,2)); rho(1,1)*sqrt(rand_temp(1,1)*rand_temp(1,2)) rand_temp(1,2)];
sigma0(:, :) = [rand_temp(2,1) rho(1,2)*sqrt(rand_temp(2,1)*rand_temp(2,2)); rho(1,2)*sqrt(rand_temp(2,1)*rand_temp(2,2)) rand_temp(2,2)];

result = zeros(10, 2);                  % 输出结果，[循环次数, 对应的\phi]
last_phi = 1;                           % matlab 没有do-while循环，设置为1，好满足第一次判断条件
k = 1;                                  % 记录迭代次数
w_i = zeros(samples, 1);                % 每个样本都有一个w(i)，共20000个
while (abs(last_phi-phi(1, 1)) > 10^(-6))   % 这里要用绝对值，曲线是上升还是下降，不可预知
    disp(last_phi-phi(1, 1));               % 观察误差
    last_phi = phi(1, 1);
     
    result(k,:)  = [k,phi(1, 1)];           % 保存结果，观察每次的phi取值变化
    k = k+1;                                % 迭代次数
    
    % 求取 w(i) = p_x_z1/(p_x_z1+p_x_z0)
    % p_x_z1 <- [P{X|z=1} ~ N(mu1, sigma1)]*phi
    % p_x_z0 <- [P{X|z=0} ~ N(mu0, sigma0)]*(1-phi)
    % 更新w(i)，共20000个
    for i = 1:samples
        X_Y = train_data(i, :)';  
        % 下面两种方式等价-计算概率
        % 方式一
        temp1 = X_Y - mu(:,1,1);        % 列向量2*1 -> x^(i)-mu1
        temp0 = X_Y - mu(:,1,2);        % 列向量2*1 -> x^(i)-mu0
        p_x_z1 = 1/(2*pi*sqrt(det(sigma1)))*exp(-0.5*temp1'*inv(sigma1)*temp1)*phi(1,1);
        p_x_z0 = 1/(2*pi*sqrt(det(sigma0)))*exp(-0.5*temp0'*inv(sigma0)*temp0)*phi(1,2);
        
%         % 方式二 - 好像很慢啊，不是太好，不太建议
%         p_x_z1 = mvnpdf(X_Y, mu(:,1,1), sigma1);
%         p_x_z0 = mvnpdf(X_Y, mu(:,1,2), sigma0);
        w_i(i, 1) = p_x_z1 / (p_x_z0 + p_x_z1);     % 注意：假如sigma1/sigma0四个元素随机初始化，有可能出现1-rho^2小于0，开方成复数，导致错误
    end
    
    phi(1, 1) = sum(w_i)/samples; phi(1, 2) = 1 - phi(1, 1);                                            % 更新phi及1-phi
    for j = 1:2
        mu(j, 1, 1) = sum(w_i(:, 1).*train_data(:, j))/sum(w_i);                                        % 更新mu1
        mu(j, 1, 2) = sum((1-w_i(:, 1)).*train_data(:, j))/(samples-sum(w_i));                          % 更新mu0
    end
    
    % 上面和下面分别是20000个循环，如果可以合并，将大大提升效率，然而不能合并，因为要先计算mu1/mu0,才能计算sigma1/sigma0
    sigma1 = 0;
    sigma0 = 0;
    for i = 1:samples
       sigma1  = sigma1 + w_i(i, 1)*(train_data(i, :)-mu(:,1,1)')'*(train_data(i, :)-mu(:,1,1)');       % 更新矩阵sigma1
       sigma0  = sigma0 + (1-w_i(i, 1))*(train_data(i, :)-mu(:,1,2)')'*(train_data(i, :)-mu(:,1,2)');   % 更新矩阵sigma0
    end
    sigma1 = sigma1/sum(w_i);
    sigma0 = sigma0/(samples-sum(w_i));                                                                 % 把1从求和符号拿出来，提高效率
    
end

num2str(result(:, 2), '%.30g');                 % 提高保存精度
position = [char(times+65), num2str(1)];        % 获取excl的存放位置，如'A1','C1','E1',..
xlswrite('result.xls',  result, 1, position);   % sheet1，position-按'A1','C1','E1',..存放数据

end

