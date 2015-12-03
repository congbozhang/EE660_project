clear;
clc;
close all;

input=load('train_zcb_preprocessed.csv');

y_col=34;
cat_fea_col=[6,7,8,9,10,11,13,14,17,18,19,22,24,30,31,32];
cat_fea_dim=[8,10,2,4,4,6,12,4,8,18,2,2,6,2,2,5];

x=input;
x(:,[cat_fea_col,y_col])=[];

%categorical features binarization
for i=1:size(cat_fea_col,2)
   temp=zeros(size(input,1),cat_fea_dim(i));
   for j=1:size(input,1)
       temp(j,input(j,cat_fea_col(i)))=temp(j,input(j,cat_fea_col(i)))+1;  
   end
   x=[x,temp];
end
y=input(:,y_col);

%the first column is num , so drop it
x(:,1)=[];

%splitting data into training set and testing set
num_train=int32(size(x,1)/2);
num_test=size(x,1)-num_train;
indices=randperm(num_train);
xtrain=x(indices,:);
xtest=x;
xtest(indices,:)=[];
ytrain=y(indices,:);
ytest=y;
ytest(indices,:)=[];

%normalization
mean_train=mean(xtrain);
std_train=std(xtrain);
std_train(std_train==0)=0.001;
xtrain=(xtrain-repmat(mean_train,size(xtrain,1),1))./repmat(std_train,size(xtrain,1),1);
xtest=(xtest-repmat(mean_train,size(xtest,1),1))./repmat(std_train,size(xtest,1),1);

save('C:\Users\Congbo\Desktop\EE660_project\data.mat','xtrain','xtest','ytrain','ytest');
