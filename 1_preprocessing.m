clear;
clc;
close all;

input=load('train_zcb_preprocessed.csv');

y_col=2;
cat_fea_col=[6,7,8,9,10,11,13,14,17,18,19,22,24,30,31,32];
cat_fea_dim=[8,10,2,4,4,6,12,4,8,18,2,2,6,2,2,5];

x=input;
x(:,[cat_fea_col,y_col])=[];
y=input(:,y_col);

%categorical features binarization
for i=1:size(cat_fea_col,2)
   temp=zeros(size(input,1),cat_fea_dim(i));
   for j=1:size(input,1)
       temp(j,input(j,cat_fea_col(i)))=temp(j,input(j,cat_fea_col(i)))+1;  
   end
   x=[x,temp];
end

%the first column is sample num , so drop it
x(:,1)=[];

%splitting data into training set , validation set and testing set
num_train=int32(size(x,1)/3);
num_val=int32(size(x,1)/3);
num_test=size(x,1)-num_train-num_val;

randindex=randperm(size(x,1));
index_train=randindex(1:num_train);
index_val=randindex(num_train+1:num_train+num_val);
index_test=randindex(num_train+num_val+1:end);

xtrain=x(index_train,:);
xval=x(index_val,:);
xtest=x(index_test,:);
ytrain=y(index_train,:);
yval=y(index_val,:);
ytest=y(index_test,:);

%normalization
mean_train=mean([xtrain;xval]);
std_train=std([xtrain;xval]);
xtrain=(xtrain-repmat(mean_train,size(xtrain,1),1))./repmat(std_train,size(xtrain,1),1);
xval=(xval-repmat(mean_train,size(xval,1),1))./repmat(std_train,size(xval,1),1);
xtest=(xtest-repmat(mean_train,size(xtest,1),1))./repmat(std_train,size(xtest,1),1);

%save('C:\Users\Congbo\Desktop\EE660_project\data.mat','xtrain','xval','xtest','ytrain','yval','ytest');
