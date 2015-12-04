clear;
clc;
close all;

load data.mat

t=dtfit(xtrain,ytrain,'forceRegression',true);

yhattrain=dtpredict(t,xtrain);
yhattest=dtpredict(t,xtest);

msetrain=sum((yhattrain-ytrain).^2)./size(ytrain,1);
msetest=sum((yhattest-ytest).^2)./size(ytest,1);