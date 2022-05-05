
%% �ô���Ϊ�����Ŵ��㷨�������Ԥ�����
% ��ջ�������
clc
clear
% 

%% ���ݶ�ȡ
% �� Excel ����ļ� myData.xlsx �ж�����
p = xlsread('myData',1,'C:G');%�Ͻ�Ԫ�غ�����%����ע�¶ȣ��棩	�����ٶȣ�mm/min��
t = xlsread('myData',1,'H:H');%������С���0�8m��
%����и����� ͷ������min��max
pmin = p(1,:);
pmax = p(2,:);
tmin = t(1,:);
tmax = t(2,:);
p(1:2,:) = [];  % ɾ���� 1��2 �� ��������max���ݣ�����ʵ������
t(1:2,:) = [];  % ɾ���� 1��2 ��
% ��һ������
p = ((p-pmin)./(pmax-pmin))';
t = ((t-tmin)./(tmax-tmin))';

%�������̻����� std ȱʧֵ �쳣ֵ
%mapminmax(); 'reverse'������

%���������ļ����Ż�����
% tmin ֮���Ӧ�Ĺ�һ��ֵ��������
tmin_hf_1 = 0.85*tmin;
tmin_hf_1 = ((tmin_hf_1-tmin)./(tmax-tmin))';
% tmax ֮ 10 ����Ӧ�Ĺ�һ��ֵ��������
tmax_x10 = 10*tmax;
tmax_x10 = ((tmax_x10-tmin)./(tmax-tmin))';

%% ����ṹ����
%��ȡ����
load data input output

% %�ڵ����
inputnum=5;
hiddennum=[20 12 8];
outputnum=1;

%--------------------------------------------------------------------------
S(1) = 20;           % �� 1 �������Ԫ��
S(2) = 12;           % �� 2 �������Ԫ��
S(3) = 8;            % �� 3 �������Ԫ��
net = feedforwardnet(S);    % �������� trainFcn = default = 'trainlm'
% �������롢������ݽṹ�������������������Ԫ��
net = configure(net,p,t);
view(net);

%ѵ�����ݺ�Ԥ������
input_train=p(:,1:400);%ȡ1��1900�е�������ѵ��
input_test=p(:,401:end);%ȡ1901��2000�е�����������
output_train=t(:,1:400);
output_test=t(:,401:end);

% %ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);%��һ����[-1,1]֮�䣬inputps��������һ��ͬ���Ĺ�һ��
[outputn,outputps]=mapminmax(output_train);
% 
% %��������
% net=newff(inputn,outputn,hiddennum);%�������㣬5����������Ԫ

%% �Ŵ��㷨������ʼ��
maxgen=20;                          %��������������������
sizepop=10;                         %��Ⱥ��ģ
pcross=0.2;                       %�������ѡ��0��1֮��
pmutation=0.1;                    %�������ѡ��0��1֮��

%�ڵ�����������������Ȩֵ��������ֵ�����������Ȩֵ�������ֵ��4���������һ��Ⱦɫ�壩
% numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;%21����10,5,5,1
numsum = net.numWeightElements;
lenchrom=ones(1,numsum);%���峤�ȣ���ʱ�����ΪȾɫ�峤�ȣ���1��numsum�еľ���      
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %��numsum��2�еĴ������󣬵�1����-3����2����3

%------------------------------------------------------��Ⱥ��ʼ��--------------------------------------------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %����Ⱥ��Ϣ����Ϊһ���ṹ�壺10���������Ӧ��ֵ��10��Ⱦɫ�������Ϣ
avgfitness=[];                      %ÿһ����Ⱥ��ƽ����Ӧ��,һά
bestfitness=[];                     %ÿһ����Ⱥ�������Ӧ��
bestchrom=[];                       %��Ӧ����õ�Ⱦɫ�壬���������Ϣ
%��ʼ����Ⱥ
for i=1:sizepop
    %�������һ����Ⱥ
    individuals.chrom(i,:)=Code(lenchrom,bound);    %���루binary�������ƣ���grey�ı�����Ϊһ��ʵ����float�ı�����Ϊһ��ʵ��������
    x=individuals.chrom(i,:);
    %������Ӧ��
%     inputn=input_train;
%     outputn=output_train;
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   %Ⱦɫ�����Ӧ��
end

FitRecord=[];

%����õ�Ⱦɫ��
[bestfitness, bestindex]=min(individuals.fitness);%bestindex����Сֵ��������λ��/ĳ�����壩��bestfitness��ֵΪ��С��Ӧ��ֵ
bestchrom=individuals.chrom(bestindex,:);  %��õ�Ⱦɫ�壬��10����������ѡ����
avgfitness=sum(individuals.fitness)/sizepop; %Ⱦɫ���ƽ����Ӧ��(���и�����Ӧ�Ⱥ� / ������)
% ��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
trace=[avgfitness bestfitness]; %trace����1��2�У�avgfitness��bestfitness��������ֵ
 
%% ���������ѳ�ʼ��ֵ��Ȩֵ
% ������ʼ
for i=1:maxgen
    % ѡ��
    individuals=Select(individuals,sizepop); 
%     avgfitness=sum(individuals.fitness)/sizepop;%��Ⱥ��ƽ����Ӧ��ֵ
    %����
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop);
    % ����
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % ������Ӧ�� 
    for j=1:sizepop
        x=individuals.chrom(j,:); %������Ϣ
        individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);  %����ÿ���������Ӧ��ֵ 
    end
    
    %�ҵ���С�������Ӧ�ȵ�Ⱦɫ�弰��������Ⱥ�е�λ��
    [newbestfitness,newbestindex]=min(individuals.fitness);%�����Ӧ��ֵ
    [worestfitness,worestindex]=max(individuals.fitness);
    
    % ���Ÿ������
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;%ȡ�������ģ��൱����̭
    individuals.fitness(worestindex)=bestfitness;
    
    avgfitness=sum(individuals.fitness)/sizepop;
    
    trace=[trace;avgfitness bestfitness]; %��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
    FitRecord=[FitRecord;individuals.fitness];%��¼ÿһ����������Ⱥ���и������Ӧ��ֵ
end

%% �Ŵ��㷨������� 
figure(1)
[r, c]=size(trace);
plot([1:r]',trace(:,2),'b--');
title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
xlabel('��������');ylabel('��Ӧ��');
legend('ƽ����Ӧ��','�����Ӧ��');
disp('��Ӧ��                   ����');

%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��
% w1=x(1:inputnum*hiddennum);
% B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
% w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
% B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
% 
% net.iw{1,1}=reshape(w1,hiddennum,inputnum);
% net.lw{2,1}=reshape(w2,outputnum,hiddennum);
% net.b{1}=reshape(B1,hiddennum,1);
% net.b{2}=reshape(B2,outputnum,1);


%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��
w1=x(1:100)
b1=x(101:120)
w2=x(121:360)
b2=x(361:372)
w3=x(373:468)
b3=x(469:476)
w4=x(477:484)
b4=x(485:485)

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

%% BP����ѵ��
%�����������
net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;

%����ѵ��
[net,per2]=train(net,inputn,outputn);

%% BP����Ԥ��
%���ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=test_simu-output_test;

figure(2)
plot((output_test-test_simu)./test_simu,'-*');
title('������Ԥ�����ٷֱ�')

figure(3)
plot(error,'-*')
title('BP����Ԥ�����','fontsize',12)
ylabel('���','fontsize',12)
xlabel('����','fontsize',12)




