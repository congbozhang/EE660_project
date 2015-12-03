clear;
clc;
close all;

load data.mat

t=RegressionTree.fit(xtrain,ytrain);

yhattrain=predict(t,xtrain);
yhattest=predict(t,xtest);

msetrain=sum((yhattrain-ytrain).^2)./size(ytrain,1);
msetest=sum((yhattest-ytest).^2)./size(ytest,1);