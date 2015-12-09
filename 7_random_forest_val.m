clear;
clc;
close all;

load data_reduced.mat

rf=[9];
bs=[0.5];
nt=[50];

f=fitForest(xtrain,ytrain,'ntrees',nt,'randomFeatures',rf,'bagSize',bs);

yhat_train=predictForest(f,xtrain);
yhat_val=predictForest(f,xval);

mse_train=sum((yhat_train-ytrain).^2)./size(ytrain,1);
mse_val=sum((yhat_val-yval).^2)./size(yval,1);

xp=xval(1:5000,1:2);
yp=yhat_val(1:5000);
scatter3(xp(:,1),xp(:,2),yp);