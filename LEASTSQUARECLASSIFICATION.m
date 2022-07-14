clc;
clear;
%% importing data
firstdata=readtable('C:\Users\ASUS\Desktop\diabetes.csv');
data=table2array(firstdata);
X=data(:,1:8);
Y=data(:,9);
%% normalizing the data
 xsize=size(X);
for i=1:xsize(2)
    va=var(X(:,i));
    miu=mean(X(:,i));
    X(:,i)=(X(:,i)-miu)/sqrt(va);
end
%% finding test and train data
% generating random number for train and test data
rnd=rand(xsize(1),1);
Xrnd=[X,rnd];
Yrnd=[Y,rnd];
newX=sortrows(Xrnd,9);
newY=sortrows(Yrnd,2);
% finding test and train data
testproportion=0.8;
Xtraindata=newX(1:floor(testproportion*xsize(1)),1:xsize(2));
Ytraindata=newY(1:floor(testproportion*xsize(1)),1);
Xtestdata=newX((floor(testproportion*xsize(1))+1):end,1:xsize(2));
Ytestdata=newY((floor(testproportion*xsize(1))+1):end,1);
sizetrain=size(Xtraindata);
sizetest=size(Xtestdata);
Xtraindatawithones=[Xtraindata,ones(sizetrain(1),1)];
Xtestdatawithones=[Xtestdata,ones(sizetest(1),1)];
%% matrix T
T=[];
for i=1:sizetrain(1)
    if Ytraindata(i,1)==0
        T=[T;[1,0]];
    else
        T=[T;[0,1]];
    end
end
%% LEAST SQUARE classification
W=inv(Xtraindatawithones'*Xtraindatawithones)*Xtraindatawithones'*T;

%% evaluating the model with train data
target=W'*Xtestdatawithones';
estimation=[];
for i=1:sizetest(1)
   if target(1,i)>target(2,i)
      estimation=[estimation;0];
   else
       estimation=[estimation;1];
   end
end
%% TP/FP/TN/FN
TP=0; FP=0; TN=0; FN=0;
for i=1:sizetest(1)
  if Ytestdata(i)==estimation(i) && Ytestdata(i)==1
      TP=TP+1;
  elseif Ytestdata(i)==estimation(i) && Ytestdata(i)==0
      TN=TN+1;
  elseif Ytestdata(i)~=estimation(i) && estimation(i)==0
      FN=FN+1;
  elseif Ytestdata(i)~=estimation(i) && estimation(i)==1
      FP=FP+1;
  end
end
%% finding the number of each class in test data
testnum{1}=0;
testnum{2}=0;
for i=1:sizetest(1)
    if Ytestdata(i)==1
       testnum{2}= testnum{2}+1;
    else
       testnum{1}= testnum{1}+1;
    end
end
%% precision 
precision=TP/(TP+FP);
%% sensitivity (the proportion of positive that are classified correctly)
sensitivity=TP/testnum{2};
%% accurracy (the proportion of all test data which is classified correctly)
accuracy=(TP+TN)/(testnum{1}+testnum{2});
%% specificity (the proportion of negative that are classified correctly)
specificity=TN/testnum{1};
%% displaying outputs
fprintf('precision: %f\n',precision)
fprintf('sensitivity: %f\n',sensitivity)
fprintf('accuracy: %f\n',accuracy)
fprintf('specificity: %f\n',specificity)
