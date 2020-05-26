function  MOG(train_data, init_phi, times)
%MOG Summary of this function goes here
%   train_data: ѵ����
%   init_phi:   ʾ�����о���phi��ֵ��0.1��0.9
%   times:      ϣ��train����data,ÿ�ν��[k,phi]д��result��ͬλ�ã������꣬�����result.xls
%   ע��: ͨ������£�Ӧ����feature(��MOB��6��)�����ﲻ������X,Y(����)������feature��

samples = size(train_data,1); %features = size(train_data,2);    % 20000��������ÿ������2ά������X_Y

% phi(1:1)->phi, phi(1:2)->(1-phi)
% mu(:,1,1)->mu(1), mu(:,1,2)->mu(0)
% sigma(:,:,1)->sigma(1)-2*2, sigma(:,:,2)->sigma(0)-2*2
phi = zeros(1, 2);                           % Ԥ�ȷ��䣬Ч�ʸ�
mu = zeros(2, 1, 2);                         % ��һ������[mu(1)_X; mu(1)_Y], �ڶ�������[mu(0)_X; mu(0)_Y]
sigma1 =  zeros(2, 2);                       % sigma1,Z=1
sigma0 =  zeros(2, 2);                       % sigma0,Z=0

% ��ʼ���������sigma1/sigma0��ʼ����Ҫ�������ص㣬��֤rho����[0,1],��Ҫ�����ʼ��������w_i����ָ���
% ͨ��ʵ�飬phi������0.3102��0.6987��MOG��ʼ��ֵ�Ƚ϶࣬���深���Ķ�����ÿ��������������������ã��۲����
% ������rho��0.1��0.9������������ȡֵ����������Ըı�������ֵ����/mu1/mu0/sigma_X1/sigma_Y1/sigma_X0/sigma_Y0
phi(1, 1) = init_phi; phi(1, 2) = 1 - phi(1, 1);    % ���ȡֵ���ڶ����ǵ�һ�и���������ȡ��
mu(:, 1, 1) = [2; 6];                               % mu1��ֵ�����������[5.0; 4.99]
mu(:, 1, 2) = [1; 3];                               % mu0��ֵ�����������[0.002; 0.025]
rand_temp = [1 5; 2 6];                             % [sigma_X1 sigma_Y1; sigma_X0 sigma_Y0]
init_pho = 0.9;                                     % rho����[0,1]
rho = [init_pho, init_pho];                         % Ϊ�˼򻯣�����ʹ��z=1/z=0��rho��ͬ
sigma1(:, :) = [rand_temp(1,1) rho(1,1)*sqrt(rand_temp(1,1)*rand_temp(1,2)); rho(1,1)*sqrt(rand_temp(1,1)*rand_temp(1,2)) rand_temp(1,2)];
sigma0(:, :) = [rand_temp(2,1) rho(1,2)*sqrt(rand_temp(2,1)*rand_temp(2,2)); rho(1,2)*sqrt(rand_temp(2,1)*rand_temp(2,2)) rand_temp(2,2)];

result = zeros(10, 2);                  % ��������[ѭ������, ��Ӧ��\phi]
last_phi = 1;                           % matlab û��do-whileѭ��������Ϊ1���������һ���ж�����
k = 1;                                  % ��¼��������
w_i = zeros(samples, 1);                % ÿ����������һ��w(i)����20000��
while (abs(last_phi-phi(1, 1)) > 10^(-6))   % ����Ҫ�þ���ֵ�����������������½�������Ԥ֪
    disp(last_phi-phi(1, 1));               % �۲����
    last_phi = phi(1, 1);
     
    result(k,:)  = [k,phi(1, 1)];           % ���������۲�ÿ�ε�phiȡֵ�仯
    k = k+1;                                % ��������
    
    % ��ȡ w(i) = p_x_z1/(p_x_z1+p_x_z0)
    % p_x_z1 <- [P{X|z=1} ~ N(mu1, sigma1)]*phi
    % p_x_z0 <- [P{X|z=0} ~ N(mu0, sigma0)]*(1-phi)
    % ����w(i)����20000��
    for i = 1:samples
        X_Y = train_data(i, :)';  
        % �������ַ�ʽ�ȼ�-�������
        % ��ʽһ
        temp1 = X_Y - mu(:,1,1);        % ������2*1 -> x^(i)-mu1
        temp0 = X_Y - mu(:,1,2);        % ������2*1 -> x^(i)-mu0
        p_x_z1 = 1/(2*pi*sqrt(det(sigma1)))*exp(-0.5*temp1'*inv(sigma1)*temp1)*phi(1,1);
        p_x_z0 = 1/(2*pi*sqrt(det(sigma0)))*exp(-0.5*temp0'*inv(sigma0)*temp0)*phi(1,2);
        
%         % ��ʽ�� - ���������������̫�ã���̫����
%         p_x_z1 = mvnpdf(X_Y, mu(:,1,1), sigma1);
%         p_x_z0 = mvnpdf(X_Y, mu(:,1,2), sigma0);
        w_i(i, 1) = p_x_z1 / (p_x_z0 + p_x_z1);     % ע�⣺����sigma1/sigma0�ĸ�Ԫ�������ʼ�����п��ܳ���1-rho^2С��0�������ɸ��������´���
    end
    
    phi(1, 1) = sum(w_i)/samples; phi(1, 2) = 1 - phi(1, 1);                                            % ����phi��1-phi
    for j = 1:2
        mu(j, 1, 1) = sum(w_i(:, 1).*train_data(:, j))/sum(w_i);                                        % ����mu1
        mu(j, 1, 2) = sum((1-w_i(:, 1)).*train_data(:, j))/(samples-sum(w_i));                          % ����mu0
    end
    
    % ���������ֱ���20000��ѭ����������Ժϲ������������Ч�ʣ�Ȼ�����ܺϲ�����ΪҪ�ȼ���mu1/mu0,���ܼ���sigma1/sigma0
    sigma1 = 0;
    sigma0 = 0;
    for i = 1:samples
       sigma1  = sigma1 + w_i(i, 1)*(train_data(i, :)-mu(:,1,1)')'*(train_data(i, :)-mu(:,1,1)');       % ���¾���sigma1
       sigma0  = sigma0 + (1-w_i(i, 1))*(train_data(i, :)-mu(:,1,2)')'*(train_data(i, :)-mu(:,1,2)');   % ���¾���sigma0
    end
    sigma1 = sigma1/sum(w_i);
    sigma0 = sigma0/(samples-sum(w_i));                                                                 % ��1����ͷ����ó��������Ч��
    
end

num2str(result(:, 2), '%.30g');                 % ��߱��澫��
position = [char(times+65), num2str(1)];        % ��ȡexcl�Ĵ��λ�ã���'A1','C1','E1',..
xlswrite('result.xls',  result, 1, position);   % sheet1��position-��'A1','C1','E1',..�������

end

