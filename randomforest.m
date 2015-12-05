clear;
clc;
close all;

load data.mat

f=fitForest(xtrain,ytrain,'ntrees',10,'randomFeatures',80,'bagSize',0.9);

yhattrain=predictForest(f,xtrain);
yhattest=predictForest(f,xtest);

msetrain=sum((yhattrain-ytrain).^2)./size(ytrain,1);
msetest=sum((yhattest-ytest).^2)./size(ytest,1);