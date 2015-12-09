clear;
clc;
close all;

load data_reduced.mat

xtrain=[xtrain;xval];
ytrain=[ytrain;yval];

T=10;
K=2;
num_per_fold=int32(size(xtrain,1)/K);
fold=zeros(size(xtrain,1),1);
for i=1:K
    fold((i-1)*num_per_fold+1:i*num_per_fold)=i;
end

msetrain=0;
mseval=0;

for j=1:T
    
    randindex=randperm(size(xtrain,1));
    xfold=[xtrain(randindex,:),fold];
    yfold=[ytrain(randindex,:),fold];


    for i=1:K
        
        j
        i

        x_val_temp=xfold(xfold(:,size(xfold,2))==i,1:size(xfold,2)-1);
        x_train_temp=xfold(xfold(:,size(xfold,2))~=i,1:size(xfold,2)-1);
        y_val_temp=yfold(yfold(:,size(yfold,2))==i,1:size(yfold,2)-1);
        y_train_temp=yfold(yfold(:,size(yfold,2))~=i,1:size(yfold,2)-1);

        m=linregFit(x_train_temp,y_train_temp);

        yhattrain_temp=linregPredict(m,x_train_temp);
        yhatval_temp=linregPredict(m,x_val_temp);

        msetrain=msetrain+(1/K)*(1/T)*sum((yhattrain_temp-y_train_temp).^2)./size(y_train_temp,1);
        mseval=mseval+(1/K)*(1/T)*sum((yhatval_temp-y_val_temp).^2)./size(y_val_temp,1);

    end
end

xp=x_val_temp(1:5000,1:2);
yp=yhatval_temp(1:5000);
scatter3(xp(:,1),xp(:,2),yp);

