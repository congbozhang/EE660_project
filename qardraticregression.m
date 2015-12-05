clear;
clc;
close all;

load data_reduced2.mat

xsqrtrain=xtrain.^2;
xsqrtest=xtest.^2;

index=nchoosek([1:size(xtrain,2)],2);

xcrosstrain=[];
xcrosstest=[];
for i=1:size(index,1)
   xcrosstrain=[xcrosstrain,xtrain(:,index(i,1)).*xtrain(:,index(i,2))];
   xcrosstest=[xcrosstest,xtest(:,index(i,1)).*xtest(:,index(i,2))];
end

xtrain=[xtrain,xcrosstrain,xsqrtrain];
xtest=[xtest,xcrosstest,xsqrtest];

m=linregFit(xtrain,ytrain);

augxtrain=[ones(size(xtrain,1),1),xtrain];
augxtest=[ones(size(xtest,1),1),xtest];

yhattrain=augxtrain*m.w;
yhattest=augxtest*m.w;

msetrain=sum((yhattrain-ytrain).^2)./size(ytrain,1);
msetest=sum((yhattest-ytest).^2)./size(ytest,1);