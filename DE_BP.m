%% 该代码为基于遗传算法神经网络的预测代码
% 清空环境变量
clc
clear

%% 数据读取
% 从 Excel 表格文件 myData.xlsx 中读数据
p = xlsread('myData',1,'C:G');%合金元素含量（%）浇注温度（℃）	铸造速度（mm/min）
t = xlsread('myData',1,'H:H');%晶粒大小（µm）
%表格中给出了 头两行是min和max
pmin = p(1,:);
pmax = p(2,:);
tmin = t(1,:);
tmax = t(2,:);
p(1:2,:) = [];  % 删除第 1、2 行 是特征中max数据，不是实际数据
t(1:2,:) = [];  % 删除第 1、2 行
% 归一化处理
p = ((p-pmin)./(pmax-pmin))';
t = ((t-tmin)./(tmax-tmin))';

%特征工程还能做 std 缺失值 异常值
%mapminmax(); 'reverse'更好用

%用于其他文件中优化网络
% tmin 之半对应的归一化值，列向量
tmin_hf_1 = 0.85*tmin;
tmin_hf_1 = ((tmin_hf_1-tmin)./(tmax-tmin))';
% tmax 之 10 倍对应的归一化值，列向量
tmax_x10 = 10*tmax;
tmax_x10 = ((tmax_x10-tmin)./(tmax-tmin))';

%% 网络结构建立
%读取数据
load data input output
%--------------------------------------------------------------------------
S(1) = 20;           % 第 1 隐层的神经元数
S(2) = 12;           % 第 2 隐层的神经元数
S(3) = 8;            % 第 3 隐层的神经元数
net = feedforwardnet(S);    % 建立网络 trainFcn = default = 'trainlm'
% 根据输入、输出数据结构，配置输入和输出层的神经元数
net = configure(net,p,t);
view(net);

%训练数据和预测数据
input_train=p(:,1:400);%取1到1900行的数据来训练
input_test=p(:,401:end);%取1901到2000行的数据来测试
output_train=t(:,1:400);
output_test=t(:,401:end);

% %选连样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);%归一化到[-1,1]之间，inputps用来作下一次同样的归一化
[outputn,outputps]=mapminmax(output_train);
[input_test,input_testps]=mapminmax(input_test);
[output_test,output_testps]=mapminmax(output_test);
%
test_input = input_test;

train_input = inputn;
train_output =outputn;
%
x_train = inputn;
x_test = input_test;
y_train = outputn;
y_test = output_test;


%% 设置DE参数
sizepop = 20;                      %种群规模
dim = net.numWeightElements;       %优化参数
lb = [-2];ub = [2];                %上下界
mutation = 0.5;                     %变异概率选择，0和1之间
crossover = 0.2;                    %交叉概率选择，0和1之间
maxgen = 50;                        %进化代数，即迭代次数

if size(lb,1) == 1
    lb = ones(dim,1).*lb;
    ub = ones(dim,1).*ub;
end

%% 个体初始化

Targetfitness = inf;
for i = 1:sizepop
    position(i,:) = lb'+(ub'-lb').*rand(1,dim); % 随机初始化个体  
    
    predict = DE_fit(net,train_input,train_output,position(i,:));% 计算个体目标函数值
    predict = mapminmax('reverse',predict,outputps);
    fit(i) = sqrt(sum((predict'-y_train).^2)/length(y_train));
    if fit(i) < Targetfitness  % 如果个体目标函数值优于当前最优值
        Targetfitness = fit(i);% 更新最优值
        Targetposition = position(i,:);%保留
    end
end

%% 迭代寻优
converage = zeros(1,maxgen);% 初始化迭代最优值
for L = 1:maxgen
    for j = 1:sizepop% 遍历每个个体
        %变异操作
        ri = randperm(sizepop,3); % 随机选择三个个体以备变异使用
        while isempty(find(ri==j))==0
            ri = randperm(sizepop,3);
        end
        mpop = position(ri(1),:)+mutation*(position(ri(2),:)-position(ri(3),:));%差分向量
        if isempty(find(mpop>ub'))== 0
            mpop(find(mpop>ub')) = (ub(find(mpop>ub')))';
        end
        if isempty(find(mpop<lb'))== 0
            mpop(find(mpop<lb')) = (lb(find(mpop<lb')))';
        end
        
        %交叉操作
        tmp = zeros(1,dim);
        for i = 1:dim
            if rand < crossover
                tmp(i) = mpop(i);
            else
                tmp(i) = position(j,i);
            end
        end
        
        %选择操作 
        cpredict = DE_fit(net,train_input,train_output,tmp);
        cpredict = mapminmax('reverse',cpredict,outputps);
        cfit = sqrt(sum((cpredict'-y_train).^2)/length(y_train));
        

        %更新最优
        if cfit < fit(j)
            position(j,:) = tmp;
            fit(j) = cfit;
        end
        if fit(j) < Targetfitness
            Targetposition = position(j,:);
            Targetfitness = fit(j);
        end    
    end
    converage(L) = Targetfitness;% 保存当前迭代最优个体函数值 Update Best Cost
end
%% 图示寻优过程
plot(converage);
xlabel('Iteration');
ylabel('Best Val');
grid on;
%% 把最优初始阀值权值赋予网络预测
% %用遗传算法优化的BP网络进行值预测
x = position(j,:);
w1=x(1:100)
b1=x(101:120)
w2=x(121:360)
b2=x(361:372)
w3=x(373:468)
b3=x(469:476)
w4=x(477:484)
b4=x(485:485)

%% 使用优化的参数进行训练
%input weight
net.iw{1,1}=reshape(w1,20,5);%将w1由1行inputnum*hiddennum列转为hiddennum行inputnum列的二维矩阵
%layer weight
net.lw{2,1}=reshape(w2,[12 20]);%更改矩阵的保存格式
net.lw{3,2}=reshape(w3,8,12);
net.lw{4,3}=reshape(w4,1,8);
net.b{1}=reshape(b1,20,1);%1行hiddennum列，为隐含层的神经元阈值
net.b{2}=reshape(b2,12,1);
net.b{3}=reshape(b3,8,1);

net.trainParam.epochs = 1000;
net.trainParam.lr = 0.01;
net.trainParam.goal = 0.00001;

net = train(net,train_input,train_output);

save('net');
%% 测试效果
predict = sim(net,train_input);
predict_train = mapminmax('reverse',predict,outputps);
train_rmse = sqrt(sum((predict_train'-y_train).^2)/length(y_train));
predict = sim(net,test_input);
predict_test = mapminmax('reverse',predict,outputps);
test_rmse = sqrt(sum((predict_test'-y_test).^2)/length(y_test));
disp(['Train RMSE = ',num2str(train_rmse),' Test RMSE = ',num2str(test_rmse)])
save('test_rmse');
%数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=test_simu-output_test;

figure(2)
plot((output_test-test_simu)./test_simu,'-*');
title('神经网络预测误差百分比')

figure(3)
plot(error,'-*')
title('BP网络预测误差','fontsize',12)
ylabel('误差','fontsize',12)
xlabel('样本','fontsize',12)


