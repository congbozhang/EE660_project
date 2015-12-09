clear;
clc;
close all;

load data_reduced.mat

rf=[9];
bs=[0.5];
nt=[50];

k=5;
num_per_fold=int32(size(xtrain,1)/k);
fold=zeros(size(xtrain,1),1);
for i=1:k
    fold((i-1)*num_per_fold+1:i*num_per_fold)=i;
end

randindex=randperm(size(xtrain,1));
xfold=[xtrain(randindex,:),fold];
yfold=[ytrain(randindex,:),fold];

mse_train_cv=zeros(size(rf,2),size(bs,2),size(nt,2));
mse_val_cv=zeros(size(rf,2),size(bs,2),size(nt,2));

for i=1:k
    
    x_val_temp=xfold(xfold(:,size(xfold,2))==i,1:size(xfold,2)-1);
    x_train_temp=xfold(xfold(:,size(xfold,2))~=i,1:size(xfold,2)-1);
    y_val_temp=yfold(yfold(:,size(yfold,2))==i,1:size(yfold,2)-1);
    y_train_temp=yfold(yfold(:,size(yfold,2))~=i,1:size(yfold,2)-1);
    
    for j=1:size(rf,2)
        for l=1:size(bs,2)
            for m=1:size(nt,2)
            
                i
                j
                l
                m

                f=fitForest(x_train_temp,y_train_temp,'ntrees',nt(m),'randomFeatures',rf(j),'bagSize',bs(l));

                yhat_train_temp=predictForest(f,x_train_temp);
                yhat_val_temp=predictForest(f,x_val_temp);

                mse_train_cv(j,l,m)=mse_train_cv(j,l,m)+(1/k)*sum((yhat_train_temp-y_train_temp).^2)./size(y_train_temp,1);
                mse_val_cv(j,l,m)=mse_val_cv(j,l,m)+(1/k)*sum((yhat_val_temp-y_val_temp).^2)./size(y_val_temp,1);
            end
        end
    end
end