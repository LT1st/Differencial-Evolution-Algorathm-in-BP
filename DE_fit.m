function error = DE_fit(net,train_input,train_output,tmp)
%该函数用来计算适应度值
%x          input     染色体信息
%inputnum   input     输入层节点数
%outputnum  input     隐含层节点数
%net        input     网络
%inputn     input     训练输入数据
%outputn    input     训练输出数据
%error      output    个体适应度值
x=tmp;
inputn = train_input;
outputn = train_output;
%提取
w1=x(1:100)
b1=x(101:120)
w2=x(121:360)
b2=x(361:372)
w3=x(373:468)
b3=x(469:476)
w4=x(477:484)
b4=x(485:485)
%----------------------------------------------------------
%net.biases{1}   net.inputWeights{1} net.layerWeights
%https://blog.csdn.net/weixin_44803715/article/details/116292744
%net.layerConnect 层间连接 12 23 34

%网络进化参数
net.trainParam.epochs=20;%迭代次数
net.trainParam.lr=0.1;%学习率
net.trainParam.goal=0.00001;%最小目标值误差
net.trainParam.show=100;
net.trainParam.showWindow=0;
 
%网络权值赋值
%input weight
net.iw{1,1}=reshape(w1,20,5);%将w1由1行inputnum*hiddennum列转为hiddennum行inputnum列的二维矩阵
%layer weight
net.lw{2,1}=reshape(w2,[12 20]);%更改矩阵的保存格式
net.lw{3,2}=reshape(w3,8,12);
net.lw{4,3}=reshape(w4,1,8);
net.b{1}=reshape(b1,20,1);%1行hiddennum列，为隐含层的神经元阈值
net.b{2}=reshape(b2,12,1);
net.b{3}=reshape(b3,8,1);

%网络训练
net=train(net,inputn,outputn);

an=sim(net,inputn);
% 以曼哈顿距离作为优化的目标函数值
error=sum(abs(an-outputn));%染色体对应的适应度值