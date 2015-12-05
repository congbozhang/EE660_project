clear;
clc;
close all;

load data.mat

lambda=71:2:91;

msetrain=zeros(1,size(lambda,2));
msetest=msetrain;

for i=1:size(lambda,2)
    i
    
    m=linregFit(xtrain,ytrain,'lambda',lambda(i),'regType','L1');

    augxtrain=[ones(size(xtrain,1),1),xtrain];
    augxtest=[ones(size(xtest,1),1),xtest];

    yhattrain=augxtrain*m.w;
    yhattest=augxtest*m.w;

    msetrain(i)=sum((yhattrain-ytrain).^2)./size(ytrain,1);
    msetest(i)=sum((yhattest-ytest).^2)./size(ytest,1);
    
end

p=find(msetest==min(msetest));

m=linregFit(xtrain,ytrain,'lambda',lambda(p),'regType','L1');

weight=abs(m.w(2:end,:));

index=[];
for i=1:size(weight,1)
    if weight(i)>=0.1*(max(weight))
        index=[index;i];
    end
end

xtrain=xtrain(:,index);
xtest=xtest(:,index);

save('C:\Users\Congbo\Desktop\EE660_project\data_reduced2.mat','xtrain','xtest','ytrain','ytest');
