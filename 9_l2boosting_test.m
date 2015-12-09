clear;
clc;
close all;

load data_reduced.mat

M=487;
v=0.1;

xtrain=[xtrain;xval];
ytrain=[ytrain;yval];

msetrain=zeros(1,M+1);
msetest=msetrain;

yhattrain=zeros(size(xtrain,1),M+1);
yhattest=zeros(size(xtest,1),M+1);

t=dtfit(xtrain,ytrain,'forceRegression',true,'maxdepth',2);

yhattrain(:,1)=dtpredict(t,xtrain);
yhattest(:,1)=dtpredict(t,xtest);

msetrain(1)=sum((yhattrain(:,1)-ytrain).^2)./size(ytrain,1);
msetest(1)=sum((yhattest(:,1)-ytest).^2)./size(ytest,1);

for m=1:M
    
    m
    
    residue=ytrain-yhattrain(:,m);
    
    t=dtfit(xtrain,residue,'forceRegression',true,'maxdepth',2);
    
    yhattrain_temp=dtpredict(t,xtrain);
    yhattest_temp=dtpredict(t,xtest);
    
    yhattrain(:,m+1)=yhattrain(:,m)+yhattrain_temp.*v;
    yhattest(:,m+1)=yhattest(:,m)+yhattest_temp.*v;
    
    msetrain(m+1)=sum((yhattrain(:,m+1)-ytrain).^2)./size(ytrain,1);
    msetest(m+1)=sum((yhattest(:,m+1)-ytest).^2)./size(ytest,1);
    
end
