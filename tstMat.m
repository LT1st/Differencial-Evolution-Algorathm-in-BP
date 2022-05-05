load('net')
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

figure(4)
plot(train_rmse,'-*')
title('MSE','fontsize',12)
ylabel('RMSE','fontsize',12)
xlabel('样本','fontsize',12)