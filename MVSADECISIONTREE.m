%% train
clc;
clear;
close all;
firstdata=readtable('C:\Users\ASUS\Desktop\glass.csv');
data=table2array(firstdata);
target=data(:,11);
[datasizer datasizec]=size(data);
variables=data(:,2:10);
numberofclass=unique(target);
[r c]=size(numberofclass);
%% normalizing the data
s=size(variables);
for i=1:s(2)
    covmtx=var(variables(:,i));
    miu=mean(variables(:,i));
    variables(:,i)=(variables(:,i)-miu)/sqrt(covmtx);
end
data=variables;
%% PCA
%% finding covariance matrix
covmtx=cov(variables);
%% finding eigen values and eigen vectors of cov matrix
[eigvec eigval]=eig(covmtx);
% it is clear that S=eigvector*eigenvalue*eigenvector'
%% finding necessary eigen values threshold=0.9
sizeig=size(eigval);
bank=0;
nofnecessaryeig=0;
for p=1:sizeig(1)
    bank=bank+eigval(sizeig-p+1,sizeig-p+1);
    nofnecessaryeig=nofnecessaryeig+1;
    if bank/sum(sum(eigval))>0.85
        break;
    end
end
%% creating new data from our main data
newU=eigvec(:,sizeig-nofnecessaryeig+1:sizeig);
newdataPca=variables*newU;

%% separating test and train data
% generating random number for train and test data
xsize=size(newdataPca);
rnd=rand(xsize(1),1);
Xrnd=[newdataPca,rnd];
Yrnd=[target,rnd];
newX=sortrows(Xrnd,xsize(2)+1);
newY=sortrows(Yrnd,2);
% splitting data into test and train
trainproportion=0.7;
trainX=newX(1:ceil(trainproportion*xsize(1)),1:nofnecessaryeig);
testX=newX((ceil(trainproportion*xsize(1))+1):end,1:nofnecessaryeig);
trainY=newY(1:ceil(trainproportion*xsize(1)),1);
testY=newY((ceil(trainproportion*xsize(1))+1):end,1);
sizetrain=size(trainX);
sizetest=size(testX);

%% input part for type of classifier
classifier=questdlg('Which method do you prefer?','method selection','decision Tree', 'Knn' , 'Other methods', 'decision Tree');
%% classifier
%%%%%%%%%%%%%%%%%%%%%%%%%
if classifier==string('Other methods')
    classifier=questdlg('Which method do you prefer?','method selection','Least square', 'Fisher' , 'cancel', 'Least square');
end
if classifier==string('decision Tree')
    %% creating decisiontree
    tree=fitctree(trainX,trainY,'MaxNumSplits',15);
    view(tree,'Mode','graph')
    estimation=ones(sizetest(1),1);
    for i=1:sizetest(1)
       estimation(i,1)= tree.predict(testX(i,:));
    end
    confmtx=confusionmat(testY',estimation');
    %% preparing testY and estimation for plotconfusion
    estimationconfmtx=zeros(sizetest(1),7);
    testconfmtx=zeros(sizetest(1),7);
    for i=1:sizetest(1)
        estimationconfmtx(i,estimation(i,1))=1;
        testconfmtx(i,testY(i,1))=1;
    end
    figure;
    plotconfusion(testconfmtx',estimationconfmtx')
    %% calculating accuracy
    testsize=size(testX);
    correct=0;
    incorrect=0;
    for i=1:testsize(1)
        if tree.predict(testX(i,:))==testY(i,1)
            correct=correct+1;
        else
            incorrect=incorrect+1;
        end
    end
    treeaccuracy=correct/(correct+incorrect)
    
    
elseif classifier==string('Knn')
    %% KNN fitting model
    Nonearestneighbers=input('enter numberof nearest neighbours: ');
    knn=fitcknn(trainX,trainY,'NumNeighbors',Nonearestneighbers);
    for i=1:sizetest(1)
       estimation(i,1)= knn.predict(testX(i,:));
    end
    confmtx=confusionmat(testY',estimation');
        %% preparing testY and estimation for plotconfusion
    estimationconfmtx=zeros(sizetest(1),7);
    testconfmtx=zeros(sizetest(1),7);
    for i=1:sizetest(1)
        estimationconfmtx(i,estimation(i,1))=1;
        testconfmtx(i,testY(i,1))=1;
    end
    figure;
    plotconfusion(testconfmtx',estimationconfmtx')
    %% calculating accuracy
    testsize=size(testX);
    correct=0;
    incorrect=0;
    for i=1:testsize(1)
        if knn.predict(testX(i,:))==testY(i,1)
            correct=correct+1;
        else
            incorrect=incorrect+1;
        end
    end
    KNNaccuracy=correct/(correct+incorrect)
elseif classifier==string('Least square')
    %% matrices with ones
    trainXwithones=[trainX,ones(sizetrain(1),1)];
    testXwithones=[testX,ones(sizetest(1),1)];
    %% matrix T
    T=zeros(sizetrain(1),r+1);
    for i=1:sizetrain(1)
       for j=1:r+1
          if trainY(i,1)==j
                  T(i,j)=1;
                  T(i,j)=1;
          end         
       end
    end
    %% W
    W=inv(trainXwithones'*trainXwithones)*trainXwithones'*T;
    %% target estimation
    testsize=size(testX);
    target=W'*testXwithones';
    estimation=[];
    for i=1:testsize(1)
        maximum=max(target(:,i));
        estimation=[estimation;find(target(:,i)==maximum)];
    end
    confmtx=confusionmat(testY',estimation');
      %% preparing testY and estimation for plotconfusion
    estimationconfmtx=zeros(sizetest(1),7);
    testconfmtx=zeros(sizetest(1),7);
    for i=1:sizetest(1)
        estimationconfmtx(i,estimation(i,1))=1;
        testconfmtx(i,testY(i,1))=1;
    end
    figure;
    plotconfusion(testconfmtx',estimationconfmtx')
    %% calculating accuracy
    correct=0;
    incorrect=0;
    for i=1:testsize(1)
        if estimation(i,1)==testY(i,1)
            correct=correct+1;
        else
            incorrect=incorrect+1;
        end
    end
    leastsquareaccuracy=correct/(correct+incorrect)
elseif classifier==string('Fisher')
    %% fisher linear discriminant
    %% finding the sample of each class in The train data
    for i=1:r+1
        classXtrain{i}=[];
        classYtrain{i}=[];
        for j=1:sizetrain(1)
           if trainY(j,1)==i
               classXtrain{i}=[classXtrain{i};trainX(j,:)];
               classYtrain{i}=[classYtrain{i};trainY(j,:)];
           end
        end
    end
    for i=1:r+1
        classXtest{i}=[];
        classYtest{i}=[];
        for j=1:sizetest(1)
           if testY(j,1)==i
               classXtest{i}=[classXtest{i};testX(j,:)];
               classYtest{i}=[classYtest{i};testY(j,:)];
           end
        end
    end
    %% finding SW
    SWTOTAL=0;
    for i=1:r+1
        if i~=4
           SW{i}=cov(classXtrain{i}); 
           SWTOTAL=SWTOTAL+SW{i};
        end 
    end
    %% mean of classes
    totalmean=mean(trainX);
    mius=[];
    for i=1:r+1
        if i~=4
           miuofclasses{i}=mean(classXtrain{i});
           mius=[mius;miuofclasses{i}-totalmean];
        end 
    end
    %% SB
    SBTOTAL=0;
        for i=1:r+1
        if i~=4
            sizeclass{i}=size(classXtrain{i});
           Sb{i}=sizeclass{i}(1,1)*miuofclasses{i}'*miuofclasses{i}; 
           SBTOTAL=SBTOTAL+Sb{i};
        end 
    end 
    %% W
    W=inv(SWTOTAL)*(SBTOTAL);
    %% reduction the dimension of W
    wcovmtx=cov(W);
    [weigvec weigval]=eig(wcovmtx);
    newW=W*weigvec(:,5);
    %% finding mean(W'X)
    for i=1:r+1
        if i~=4
          meanofclass{i}=mean((newW'*classXtrain{i}')');
        end 
    end
    %% evaluating test data
     disttomean=100000.*ones(64,7);
    for i=1:sizetest(1) 
        for j=1:r+1
            if j~=4
                datatest=classXtest{j};
                disttomean(i,j)=abs(newW'*testX(i,:)'-meanofclass{j});
            end
        end
    end
  %% finding estimation for test data
  for i=1:sizetest(1)
      min=disttomean(i,1);
      estimation(i,1)=1;
      for j=1:r+1
         if j~=4
             if disttomean(i,j)<min
                 estimation(i,1)=j;
             end
         end
      end
  end
  confmtx=confusionmat(testY',estimation');
      %% preparing testY and estimation for plotconfusion
    estimationconfmtx=zeros(sizetest(1),7);
    testconfmtx=zeros(sizetest(1),7);
    for i=1:sizetest(1)
        estimationconfmtx(i,estimation(i,1))=1;
        testconfmtx(i,testY(i,1))=1;
    end
    figure;
    plotconfusion(testconfmtx',estimationconfmtx')
        %% calculating accuracy
    correct=0;
    incorrect=0;
    for i=1:sizetest(1)
        if estimation(i,1)==testY(i,1)
            correct=correct+1;
        else
            incorrect=incorrect+1;
        end
    end
    Fisheraccuracy=correct/(correct+incorrect)
end
    
%% drawing charts
figure;
for i=1:sizetest(1)
    subplot(1,3,1)
    if estimation(i,1)==testY(i,1)
        scatter(testX(i,1),testX(i,2),'o','MarkerFaceColor',[1/estimation(i,1) 0.7 estimation(i,1)/7],'MarkerEdgeColor',[1/estimation(i,1) 0.5 1/estimation(i,1)])
        hold on;
    else
        scatter(testX(i,1),testX(i,2),'x', 'LineWidth',2,'MarkerEdgeColor',[1/testY(i,1) 0.7 testY(i,1)/7])
        hold on;
    end
    title('correct and wrong estimation (x1 vs X2)')
        subplot(1,3,2)
    if estimation(i,1)==testY(i,1)
        scatter(testX(i,3),testX(i,4),'o','MarkerFaceColor',[1/estimation(i,1) 0.7 estimation(i,1)/7],'MarkerEdgeColor',[1/estimation(i,1) 0.5 1/estimation(i,1)])
        hold on;
    else
        scatter(testX(i,3),testX(i,4),'x', 'LineWidth',2,'MarkerEdgeColor',[1/testY(i,1) 0.7 testY(i,1)/7])
        hold on;
    end
    title('correct and wrong estimation (x3 vs X4)')
    subplot(1,3,3)
    if estimation(i,1)==testY(i,1)
        scatter(testX(i,4),testX(i,5),'o','MarkerFaceColor',[1/estimation(i,1) 0.7 estimation(i,1)/7],'MarkerEdgeColor',[1/estimation(i,1) 0.5 1/estimation(i,1)])
        hold on; 
    else
        scatter(testX(i,4),testX(i,5),'x', 'LineWidth',2,'MarkerEdgeColor',[1/testY(i,1) 0.7 testY(i,1)/7])
        hold on;
    end
    title('correct and wrong estimation (x4 vs X5)')
end
subplot(1,3,1)
xlabel('X1')
ylabel('X2')
grid on;
subplot(1,3,2)
xlabel('X3')
ylabel('X4')
grid on;
subplot(1,3,3)
xlabel('X4')
ylabel('X5')
grid on;
hold off;