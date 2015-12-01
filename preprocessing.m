clear;
clc;
close all;

input=load('train_zcb_preprocessed.csv');

label_col=34;
cat_fea_col=[6,7,8,9,10,11,13,14,17,18,19,22,24,30,31,32];
cat_fea_dim=[8,10,2,4,4,6,12,4,8,18,2,2,6,2,2,5];

features=input;
features(:,[cat_fea_col,label_col])=[];

%categorical features binarization
for i=1:size(cat_fea_col,2)
   temp=zeros(size(input,1),cat_fea_dim(i));
   for j=1:size(input,1)
       temp(j,input(j,cat_fea_col(i)))=temp(j,input(j,cat_fea_col(i)))+1;  
   end
   features=[features,temp];
end
label_train=input(:,label_col);