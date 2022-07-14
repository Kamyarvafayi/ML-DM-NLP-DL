clc;
clear;
%% initial value of variables and importing data
data = iris_dataset';
%% user set the number of clusters
n=input('enter the number of clusters(please a divisor of 150): ');
r=3; 

distancefromcenters=zeros(numel(data)/4,n);
z=[];
numerator=1;
y=linspace(1,numel(data)/4,numel(data)/4);
%% setting memberships randomly
membership=zeros(numel(data)/4,n);
%     randomnumbers=rand(n,1);
    for i=1:n
        for e=((150/n)*(i-1)+1):(150/n)*i
%         if randomnumbers(i,1)==max(randomnumbers)
          membership(e,i)=1;
        end
    end
%  centers=[];
%  for k=1:n
%      p=ceil(numel(data)/4*rand());
%      centers=[centers;data(p,:)];
%  end
 
 %% FCM algorithm
% %%%%%%%%%%%%%
 while numerator<=30
      membershipinthepowerofr=membership.^r;
      %% finding centers
      centers=zeros(n,4);
           for i=1:n
              for p=1:150
                  for o=1:4
                    centers(i,o)=centers(i,o)+membershipinthepowerofr(p,i)*data(p,o)/sum(membershipinthepowerofr(:,i));
                  end
              end
           end    
     %% finding distance from centers
     distancefromcenters=zeros(numel(data)/4,n);
     for k=1:numel(data)/4
          for i=1:n
                for l=1:4
                  distancefromcenters(k,i)=distancefromcenters(k,i)+((data(k,l)-centers(i,l))^2);
                end
          end
     end
    %% calculating the value of objective function
         z(1,numerator)=0;
         for p=1:150
             for i=1:n
                   z(1,numerator)=z(1,numerator)+ membershipinthepowerofr(p,i)* distancefromcenters(p,i);
             end
         end 
   %% calculating membership 
   membership=zeros(numel(data)/4,n);
   bank=zeros(numel(data)/4,n);
   for k=1:numel(data)/4
          for i=1:n             
              if distancefromcenters(k,i)~=0
                    for t=1:n
                           bank(k,i) = bank(k,i) +((distancefromcenters(k,i)/distancefromcenters(k,t))^(1/(r-1)));
                    end
              else
                  for t=1:n
                     membership(k,t)=0;
                  end
                  membership(k,i)=1;
              end
              membership(k,i)=1/bank(k,i);
          end
   end 
if numerator>=2
     if z(1,numerator-1)-z(1,numerator)<0.000001
     break;
     end
 end
 numerator=numerator+1;

 end
%% objective function chart
 figure;
 s=size(z);
 x=linspace(1,s(1,2),s(1,2));
 line(x,z,'Color','red','marker','o')
 title('objective function value in each iteration')
 xlabel('iteration');
 ylabel('J');
 
%  %% final clusters print
%  print('the clusters are as follows');
%  disp(clusterschart) 
  %% using FCM function for comparing clusters and centers with our models
 data = iris_dataset';
 [idx,c]=fcm(data,n);