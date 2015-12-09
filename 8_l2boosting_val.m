clear;
clc;
close all;

load data_reduced.mat

M=1000;
v=0.1;

msetrain=zeros(1,M+1);
mseval=msetrain;

yhattrain=zeros(size(xtrain,1),M+1);
yhatval=zeros(size(xval,1),M+1);

t=dtfit(xtrain,ytrain,'forceRegression',true,'maxdepth',2);

yhattrain(:,1)=dtpredict(t,xtrain);
yhatval(:,1)=dtpredict(t,xval);

msetrain(1)=sum((yhattrain(:,1)-ytrain).^2)./size(ytrain,1);
mseval(1)=sum((yhatval(:,1)-yval).^2)./size(yval,1);

for m=1:M
    
    m
    
    residue=ytrain-yhattrain(:,m);
    
    t=dtfit(xtrain,residue,'forceRegression',true,'maxdepth',2);
    
    yhattrain_temp=dtpredict(t,xtrain);
    yhatval_temp=dtpredict(t,xval);
    
    yhattrain(:,m+1)=yhattrain(:,m)+yhattrain_temp.*v;
    yhatval(:,m+1)=yhatval(:,m)+yhatval_temp.*v;
    
    msetrain(m+1)=sum((yhattrain(:,m+1)-ytrain).^2)./size(ytrain,1);
    mseval(m+1)=sum((yhatval(:,m+1)-yval).^2)./size(yval,1);
    
end
