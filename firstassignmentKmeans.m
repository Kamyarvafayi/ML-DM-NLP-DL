clc;
clear;
%% initial value of variables and importing data
data = iris_dataset';
%% user set the number of clusters
n=input('enter the number of clusters: ');
clusters=zeros(numel(data)/4,n);
distancefromcenters=zeros(numel(data)/4,n);
z=[];
numerator=1;
y=linspace(1,numel(data)/4,numel(data)/4);
%% setting centers randomly
 centers=[];
for k=1:n
    p=ceil(numel(data)/4*rand());
    centers=[centers;data(p,:)];
end
% for i=1:n
%     for j=1:4
%         centers(i,j)=(sum(data(((150/n)*i-(150/n)+1):((150/n)*i),j)))/(150/n);
%     end
% end


%% kmeans algorithm
%%%%%%%%%%%
while numerator<=8
    %% finding distance from centers and clustering
    distancefromcenters=zeros(numel(data)/4,n);
    clusterschart=zeros(150,1);
    for k=1:numel(data)/4
         for i=1:n
               for l=1:4
                 distancefromcenters(k,i)=distancefromcenters(k,i)+((data(k,l)-centers(i,l))^2);
               end
         end
         for i=1:n
               if min(distancefromcenters(k,:))==distancefromcenters(k,i)
                        clusters(k,i)=1;
               else 
                        clusters(k,i)=0;
               end
         end
         for i=1:n
             clusterschart(k,1)=clusterschart(k,1)+clusters(k,i)*i;
         end
    end
    %% calculating the value of objective function
          z(1,numerator)=0;
         for p=1:150
             for i=1:n
                   z(1,numerator)=z(1,numerator)+clusters(p,i)* distancefromcenters(p,i);
             end
         end 
         %% changing centers
          centers=zeros(n,4);
          for o=1:4
             for p=1:150
                 for i=1:n
                   centers(i,o)=centers(i,o)+clusters(p,i)*data(p,o)/sum(clusters(:,i));
                 end
             end
          end
    %% figures
    %% x1 vs x2 charts
    if numerator==1
    figure;
    end
    subplot(4,4,numerator)
    hold on;
    for k=1:150
        scatter(data(k,1),data(k,2),'MarkerEdgeColor',[0 clusterschart(k,1)/n clusterschart(k,1)/n], 'MarkerFaceColor',[0 clusterschart(k,1)/n clusterschart(k,1)/n])
%         
    end
    axis([1 8 1 8])
    title('x1 vs x2')
    xlabel('X1');
    ylabel('X2');
    %% centers in charts x1 vs x2
    for i=1:n
        scatter(centers(i,1),centers(i,2),'marker','s','LineWidth',7)
    end
    
    axis([1 8 1 8])
    hold off;
    %% x3 vs x4 charts
    subplot(4,4,numerator+8)
    hold on;
    for k=1:150
        scatter(data(k,3),data(k,4),'MarkerEdgeColor',[0 clusterschart(k,1)/n clusterschart(k,1)/n], 'MarkerFaceColor',[0 clusterschart(k,1)/n clusterschart(k,1)/n])
       
    end
    axis([0 7 0 7])
    title('x3 vs x4')
    xlabel('X3');
    ylabel('X4');
    
    %% centers in charts x3 vs x4
    for i=1:n
        scatter(centers(i,3),centers(i,4),'marker','s','LineWidth',7)
    end
    axis([0 7 0 7])
    hold off;
    %% convergence criteria
    if numerator>=2
     if z(1,numerator-1)-z(1,numerator)<0.001
     break;
     end
 end
      numerator=numerator+1;
end

%% objective function chart
 figure;
 sz=size(z);
 x=linspace(1,sz(1,2),sz(1,2));
 line(x,z,'Color','red','marker','o')
 title('objective function value in each iteration')
 xlabel('iteration');
 ylabel('J');
 
 %% final clusters print
 print('the clusters are as follows');
 disp(clusterschart')
 
 %% using kmeans function for comparing clusters and centers
 [idx,c]=kmeans(data,n);
 