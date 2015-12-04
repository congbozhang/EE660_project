clear;
clc;
close all;

load data.mat

m=linregFit(xtrain,ytrain, 'lambda',5000,'regType', 'L1');

augxtrain=[ones(size(xtrain,1),1),xtrain];
augxtest=[ones(size(xtest,1),1),xtest];

yhattrain=augxtrain*m.w;
yhattest=augxtest*m.w;

msetrain=sum((yhattrain-ytrain).^2)./size(ytrain,1);
msetest=sum((yhattest-ytest).^2)./size(ytest,1);

stem(m.w);