clear;
clc;
close all;

load data.mat

lambda=[0.2,0.4,0.6,0.8,1,1.5,2,3,5];

msetrain=zeros(1,size(lambda,2));
msetest=msetrain;

for i=1:size(lambda,2)
    i
    
    m=linregFit(xtrain,ytrain,'lambda',lambda(i));

    augxtrain=[ones(size(xtrain,1),1),xtrain];
    augxtest=[ones(size(xtest,1),1),xtest];

    yhattrain=augxtrain*m.w;
    yhattest=augxtest*m.w;

    msetrain(i)=sum((yhattrain-ytrain).^2)./size(ytrain,1);
    msetest(i)=sum((yhattest-ytest).^2)./size(ytest,1);
    
end