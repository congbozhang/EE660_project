clear;
clc;
close all;

load data.mat

lambda=150:4:250;

msetrain=zeros(1,size(lambda,2));
mseval=msetrain;

for i=1:size(lambda,2)
    i
    
    m=linregFit(xtrain,ytrain,'lambda',lambda(i),'regType','L1');

    augxtrain=[ones(size(xtrain,1),1),xtrain];
    augxval=[ones(size(xval,1),1),xval];

    yhattrain=augxtrain*m.w;
    yhattest=augxval*m.w;

    msetrain(i)=sum((yhattrain-ytrain).^2)./size(ytrain,1);
    mseval(i)=sum((yhattest-yval).^2)./size(yval,1);
    
end

p=find(mseval==min(mseval));

m=linregFit(xtrain,ytrain,'lambda',lambda(p),'regType','L1');

weight=abs(m.w(2:end,:));

index=[];
for i=1:size(weight,1)
    if weight(i)>=0.15*(max(weight))
        index=[index;i];
    end
end

xtrain=xtrain(:,index);
xval=xval(:,index);
xtest=xtest(:,index);

save('C:\Users\Congbo\Desktop\EE660_project\data_reduced.mat','xtrain','xval','xtest','ytrain','yval','ytest');
