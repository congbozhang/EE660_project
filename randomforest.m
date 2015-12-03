clear;
clc;
close all;

load data.mat

f=fitForest(xtrain,ytrain);

yhattrain=predictForest(f,xtrain);
yhattest=predictForest(f,xtest);

msetrain=sum((yhattrain-ytrain).^2)./size(ytrain,1);
msetest=sum((yhattest-ytest).^2)./size(ytest,1);