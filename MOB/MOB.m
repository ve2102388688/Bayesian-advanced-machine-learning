function  MOB(train_data, init_phi, times)
%MOB Summary of this function goes here
%   train_data: ѵ����
%   init_phi:   phi�ĳ�ʼֵ��������������0.6954��0.3047
%   times:      ϣ��train����data,ÿ�ν��[k,phi]д��result��ͬλ��
%   ע��: featureά��Ϊ6����j=1:6

samples = size(train_data,1); features = size(train_data,2); % 20000��������ÿ������6ά

% phi [phi(j11),phi(j01)] [phi(j10),phi(j00)] ����j=1:6
% phi(1:1)->phi, phi(1:2)->(1-phi)
% phi_j(:,:,1)->z=1ʱ,phi(j@1)��6*2ά����һ��@=1�� �ڶ���@=0���൱�ڵ�һ��ȡ������'@'��ʾfeature��ȡֵ��������0��1
% phi_j(:,:,2)->z=0ʱ,phi(j@1)��6*2ά����һ��@=1�� �ڶ���@=0���൱�ڵ�һ��ȡ������'@'��ʾfeature��ȡֵ��������0��1
phi = zeros(1, 2);
phi_j = zeros(6, 2, 2);                         % ��һ������[phi(j11),phi(j01)], �ڶ�������[phi(j10),phi(j00)]
% ��ʼ����ͨ��ʵ�飬���Է���phi����ֵ����0.6954��0.3047
phi(1, 1) = init_phi; phi(1, 2) = 1 - phi(1, 1);                                          % ��ʼֵ0.1��0.9
phi_j(:, 1, 1) = unifrnd(0.45, 0.55, features, 1); phi_j(:, 2, 1) = 1-phi_j(:, 1, 1);     % ����phi(j11)���ȷֲ���ʼ���������ֲ���ʼ��Ҳ����        
phi_j(:, 1, 2) = unifrnd(0.45, 0.55, features, 1); phi_j(:, 2, 2) = 1-phi_j(:, 1, 2);     % ����phi(j10)���ȷֲ���ʼ���������ֲ���ʼ��Ҳ����        

result = zeros(100, 2);     % ��������[ѭ������, ��Ӧ��\phi]
last_phi = 1;               % matlab û��do-whileѭ��������Ϊ1���������һ���ж�����
k = 1;                      % ��¼��������
w_i = zeros(samples, 1);  	% ÿ����������һ��w(i)����20000��
while (abs(last_phi-phi(1, 1)) > 10^(-6))    % ����Ҫ�þ���ֵ�����������������½�������Ԥ֪
    disp(last_phi-phi(1, 1));                % �۲����
    last_phi = phi(1, 1);
    
    result(k,:)  = [k, phi(1, 1)];           % ���������۲�ÿ�ε�phiȡֵ�仯
    k = k+1;                                 % ��������
    
    % ��ȡ w(i) = p_x_z1/(p_x_z1+p_x_z0)
    % �����һ��feature=[1 0 1 1 1 0],��p_x_z1=phi(111)*(1-phi(211))*phi(311)*phi(411)*phi(511)*(1-phi(611))*phi
    % p_x_z0=phi(110)*(1-phi(210))*phi(310)*phi(410)*phi(510)*(1-phi(610))*(1-phi)
    % ����w(i)����20000��
    for i = 1:samples
        feature = train_data(i, :)';                % ÿ��������feature
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
    
    phi(1, 1) = sum(w_i)/samples; phi(1, 2) = 1 - phi(1, 1);                                                            % ����phi��1-phi
    for j = 1:features
        phi_j(j, 1, 1) = sum(w_i(:, 1).*train_data(:, j)) / sum(w_i); phi_j(j, 2, 1) = 1-phi_j(j, 1, 1);                % ����phi_j��һ������
        phi_j(j, 1, 2) = sum((1-w_i(:, 1)).*train_data(:, j)) / (samples-sum(w_i)); phi_j(j, 2, 2) = 1-phi_j(j, 1, 2);  % ����phi_j�ڶ�������
    end
end

clear i j;                                      % �����Чѭ������
num2str(result(:, 2), '%.30g');                 % ��߱��澫��
position = [char(times+65), num2str(1)];        % ��ȡexcl�Ĵ��λ�ã���'A1','C1','E1',..
xlswrite('result.xls',  result, 1, position);   % ��'A1','C1','E1',..�������

end

