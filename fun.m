function error = fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn)
%�ú�������������Ӧ��ֵ
%x          input     Ⱦɫ����Ϣ
%inputnum   input     �����ڵ���
%outputnum  input     ������ڵ���
%net        input     ����
%inputn     input     ѵ����������
%outputn    input     ѵ���������
%error      output    ������Ӧ��ֵ

%��ȡ
% w1=x(1:inputnum*hiddennum);%ȡ������������������ӵ�Ȩֵ
% B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);%��������Ԫ��ֵ
% w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);%ȡ������������������ӵ�Ȩֵ
% B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);%�������Ԫ��ֵ
%net.
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
%net.layerConnect ������� 12 23 34

%�����������
net.trainParam.epochs=20;%��������
net.trainParam.lr=0.1;%ѧϰ��
net.trainParam.goal=0.00001;%��СĿ��ֵ���
net.trainParam.show=100;
net.trainParam.showWindow=0;
 
%����Ȩֵ��ֵ
%input weight
net.iw{1,1}=reshape(w1,20,5);%��w1��1��inputnum*hiddennum��תΪhiddennum��inputnum�еĶ�ά����
%layer weight
net.lw{2,1}=reshape(w2,[12 20]);%���ľ���ı����ʽ
net.lw{3,2}=reshape(w3,8,12);
net.lw{4,3}=reshape(w4,1,8);
net.b{1}=reshape(b1,20,1);%1��hiddennum�У�Ϊ���������Ԫ��ֵ
net.b{2}=reshape(b2,12,1);
net.b{3}=reshape(b3,8,1);

%����ѵ��
net=train(net,inputn,outputn);

an=sim(net,inputn);
error=sum(abs(an-outputn));%Ⱦɫ���Ӧ����Ӧ��ֵ

% [m n]=size(inputn);
% A1=tansig(net.iw{1,1}.*inputn+repmat(net.b{1},1,n));   %����main�����м������ͬ
% A2=purelin(net.iw{2,1}.*A1+repmat(net.b{2},1,n));      %����main�����м������ͬ  
% error=sumsqr(outputn-A2);


