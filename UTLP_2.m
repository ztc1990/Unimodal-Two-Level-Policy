price=100:100:800;%set number of cluster here
number=0.1:0.1:1;%set number of arm in each cluster
alpha_1=2;%set alpha
beta_1=2;%set beta

T=100000;%time horrizon

probability=[];
utility=[];
price_1=[];
for i=1:length(price)
    for j=1:length(number)
        probability=[probability exp(-alpha_1*(price(i)/100)-beta_1*number(j))];
        utility=[utility price(i)*exp(-alpha_1*(price(i)/100)-beta_1*number(j))];
        price_1=[price_1 price(i)];
    end
end
utility_norm=utility/max(price);
utility_max=max(utility);
gamma=2;%number of neighbors
n=length(utility);%number of arm
n_1=length(price);%number of cluster
n_2=length(number);%number of arms in each cluster

T=100000;%time horrizon

%UCB algorithm
regret_avg_UCB=[];
a_avg_UCB=[];
for j=1:50
S=zeros(1,n);
F=zeros(1,n);
a_UCB=[];%record the action for UCB
r_UCB=[];
for i=1:n
    a_UCB=[a_UCB i];
    r_UCB=[r_UCB utility_max-utility(i)];
    X = binornd(1,utility_norm(i));
    if X==1
        S(i)=S(i)+1;
    else
        F(i)=F(i)+1;
    end
end
for i=(n+1):T
    theta = S./(S+F)+sqrt((2*log(i))./(S+F));
    [~,index]=max(theta);
    a_UCB=[a_UCB index];
    r_UCB=[r_UCB utility_max-utility(index)];
    X = binornd(1,utility_norm(index));
    if X==1
        S(index)=S(index)+1;
    else
        F(index)=F(index)+1;
    end
end
regret_UCB=[];%caculate the cumulative regret
for i=1:T
    regret_UCB=[regret_UCB sum(r_UCB(1:i))];
end
regret_avg_UCB=[regret_avg_UCB;regret_UCB];
a_avg_UCB=[a_avg_UCB;a_UCB];
end

%TLP algorithm
regret_avg_TLP=[];
a_avg_TLP=[];
for j=1:50
S=zeros(1,n);
F=zeros(1,n);
S_c=zeros(1,n_1);
F_c=zeros(1,n_1);
a_TLP=[];%record the action for TS
r_TLP=[];
for i=1:n
    a_TLP=[a_TLP i];
    r_TLP=[r_TLP utility_max-utility(i)];
    X = binornd(1,utility_norm(i));
    if X==1
        S(i)=S(i)+1;
    else
        F(i)=F(i)+1;
    end
end
for i=1:n_1
    S_c(i)=sum(S(((i-1)*n_2+1):(i*n_2)));
    F_c(i)=sum(F(((i-1)*n_2+1):(i*n_2)));
end
for i=(n+1):T
    theta_c = S_c./(S_c+F_c)+sqrt((2*log(i))./(S_c+F_c));
    [~,index_c]=max(theta_c);%find cluster
    theta = S(((index_c-1)*n_2+1):(index_c*n_2))./(S(((index_c-1)*n_2+1):(index_c*n_2))+F(((index_c-1)*n_2+1):(index_c*n_2)))+sqrt((2*log(i))./(S(((index_c-1)*n_2+1):(index_c*n_2))+F(((index_c-1)*n_2+1):(index_c*n_2))));
    %betarnd(1+S(((index_c-1)*n_2+1):(index_c*n_2)),1+F(((index_c-1)*n_2+1):(index_c*n_2)));
    [~,index]=max(theta);
    action=(index_c-1)*n_2+index;
    a_TLP=[a_TLP action];
    r_TLP=[r_TLP utility_max-utility(action)];
    X = binornd(1,utility_norm(action));
    if X==1
        S(action)=S(action)+1;
        S_c(index_c)=S_c(index_c)+1;
    else
        F(action)=F(action)+1;
        F_c(index_c)=F_c(index_c)+1;
    end
end
regret_TLP=[];%caculate the cumulative regret
for i=1:T
    regret_TLP=[regret_TLP sum(r_TLP(1:i))];
end
regret_avg_TLP=[regret_avg_TLP;regret_TLP];
a_avg_TLP=[a_avg_TLP;a_TLP];
end

%UTLP algorithm
regret_avg_UTLP=[];
a_avg_UTLP=[];
for j=1:50
mu_hat=zeros(1,n);
k=zeros(1,n);
mu_c=zeros(1,n_1);
k_c=zeros(1,n_1);
l=zeros(1,n);% number of times becomes to the leader
gamma_1=gamma+1;%number of the neighbor of  the leader
a_UTLP=[];%record the action for TS
r_UTLP=[];
for i=1:n
    a_UTLP=[a_UTLP i];
    r_UTLP=[r_UTLP utility_max-utility(i)];
    %X = normrnd(mu(i),sigma_1);
    X=binornd(1,utility_norm(i));
    mu_hat(i)=(mu_hat(i)*k(i)+X)/(k(i)+1);
    k(i)=k(i)+1;
end
for i=1:n_1
    k_c(i)=sum(k(((i-1)*n_2+1):(i*n_2)));
    mu_c(i)=(sum(mu_hat(((i-1)*n_2+1):(i*n_2)).*k(((i-1)*n_2+1):(i*n_2))))/k_c(i);
end
for i=(n+1):T
    theta_c = mu_c+sqrt((2*log(i))./(k_c));
    [~,index_c]=max(theta_c);%find cluster
    mu_temp=mu_hat(((index_c-1)*n_2+1):(index_c*n_2));
    [~,leader_1]=max(mu_temp);
    leader=(index_c-1)*n_2+leader_1;
    l(leader)=l(leader)+1;
    if mod(l(leader),gamma_1)==0
        a_UTLP=[a_UTLP leader];
        r_UTLP=[r_UTLP utility_max-utility(leader)];
        %X = normrnd(mu(leader),sigma_1);
        X=binornd(1,utility_norm(leader));
        mu_hat(leader)=(mu_hat(leader)*k(leader)+X)/(k(leader)+1);
        mu_c(index_c)=(mu_c(index_c)*k_c(index_c)+X)/(k_c(index_c)+1);
        k(leader)=k(leader)+1;
        k_c(index_c)= k_c(index_c)+1;
    else
        if leader_1==1
           theta=mu_hat(leader:(leader+1))+sqrt((2*log(i))./k(leader:(leader+1)));
           %theta = normrnd(mu_hat(leader:leader+1),1./(k(leader:leader+1)));
           [~,index]=max(theta);
           a_UTLP=[a_UTLP leader-1+index];
           r_UTLP=[r_UTLP utility_max-utility(leader-1+index)];
           %X = normrnd(mu(leader-1+index),sigma_1);
           X=binornd(1,utility_norm(leader-1+index));
           mu_hat(leader-1+index)=(mu_hat(leader-1+index)*k(leader-1+index)+X)/(k(leader-1+index)+1);
           mu_c(index_c)=(mu_c(index_c)*k_c(index_c)+X)/(k_c(index_c)+1);
           k(leader-1+index)=k(leader-1+index)+1;
           k_c(index_c)= k_c(index_c)+1;
        elseif leader_1==n_2
           theta=mu_hat((leader-1):leader)+sqrt((2*log(i))./k((leader-1):leader));
           %theta = normrnd(mu_hat((leader-1):leader),1./(k((leader-1):leader)+1));
           [~,index]=max(theta);
           a_UTLP=[a_UTLP leader+index-2];
           r_UTLP=[r_UTLP utility_max-utility(leader+index-2)];
           %X = normrnd(mu(leader+index-2),sigma_1);
           X=binornd(1,utility_norm(leader+index-2));
           mu_hat(leader+index-2)=(mu_hat(leader+index-2)*k(leader+index-2)+X)/(k(leader+index-2)+1);
           mu_c(index_c)=(mu_c(index_c)*k_c(index_c)+X)/(k_c(index_c)+1);
           k(leader+index-2)=k(leader+index-2)+1;
           k_c(index_c)= k_c(index_c)+1;
        else
           theta=mu_hat(leader-1:leader+1)+sqrt((2*log(i))./k(leader-1:leader+1));
           %theta = normrnd(mu_hat(leader-1:leader+1),1./(k(leader-1:leader+1)+1));
           [~,index]=max(theta);
           a_UTLP=[a_UTLP leader+index-2];
           r_UTLP=[r_UTLP utility_max-utility(leader+index-2)];
           %X = normrnd(mu(leader+index-2),sigma_1);
           X=binornd(1,utility_norm(leader+index-2));
           mu_hat(leader+index-2)=(mu_hat(leader+index-2)*k(leader+index-2)+X)/(k(leader+index-2)+1);
           mu_c(index_c)=(mu_c(index_c)*k_c(index_c)+X)/(k_c(index_c)+1);
           k(leader+index-2)=k(leader+index-2)+1;
           k_c(index_c)= k_c(index_c)+1;
        end
    end
end
regret_UTLP=[];%caculate the cumulative regret
for i=1:T
    regret_UTLP=[regret_UTLP sum(r_UTLP(1:i))];
end
regret_avg_UTLP=[regret_avg_UTLP;regret_UTLP];
a_avg_UTLP=[a_avg_UTLP;a_UTLP];
end


set(0,'defaulttextinterpreter','latex'); % allows you to use latex math
set(0,'defaultlinelinewidth',2); % line width is set to 2
set(0,'DefaultLineMarkerSize',10); % marker size is set to 10
set(0,'DefaultTextFontSize', 16); % Font size is set to 16
set(0,'DefaultAxesFontSize',16); % font size for the axes is set to 16
figure(1)
x=1:1:T;
%plot(x,0.98*packet_num_CUUCB(1:1000), '--c',x,packet_num_CUCB(1:1000),'-*',x,packet_num_CUUCB(1:1000),'m-.+',x,0.97*packet_num_CUCB_1(1:1000),'-bo',x,packet_num_CUCB_1(1:1000),'-ro','LineWidth',2,'MarkerIndices',1:100:length(x));
%plot(x,1.2*mean(regret_co3(:,1:1000)), '--c',x,mean(regret_co2(:,1:1000)),'-*',x,mean(regret_co33(:,1:1000)),'m-.+',x,mean(regret_co22(:,1:1000)),'-bo','LineWidth',2,'MarkerIndices',1:100:length(x));
%plot(X, Y1, '-bo', X, Y2, '{rs', X); % plotting three curves Y1, Y2 for the same X
%plot(x,regret,'--c',x,regret_2,'-*',x,regret_4,'m-.+','LineWidth',2,'MarkerIndices',1:10000:length(x));
%plot(x,regret,'--c',x,regret_2,'-*',x,regret_4,'m-.+','LineWidth',2,'MarkerIndices',1:10000:length(x));
%plot(x,mean(regret_avg_TS),'r--h',x,mean(regret_avg_MTS),'g-*',x,mean(regret_avg_UTS),'-bo',x,mean(regret_avg_MUTS),'c-s',x,mean(regret_avg_UCB),'m-.+','LineWidth',2,'MarkerIndices',1:5000:length(x));
plot(x,mean(regret_avg_UCB),'g-*',x,mean(regret_avg_TLP),'c-s',x,mean(regret_avg_UTLP),'m-.+','LineWidth',2,'MarkerIndices',1:5000:length(x));
%plot(x,rate_HUUCB_1,'r--h',x,rate_HUUCB_2,'g-*','LineWidth',2,'MarkerIndices',1:5000:length(x));
grid on; % grid lines on the plot

%legend('HTS(good group)', 'HTS(Partial information)','TS','HTS(kmean)');
%legend('TS','MTS','UTS','MUTS','UCB');
legend('UCB','TLP','UTLP');
% ylabel('$Regret$ (Kbps)');
% 
% xlabel('$_$ (frames=sec)');
xlabel('Time Slot');
ylabel('Regret');
title('Regret vs Time');
set(0,'defaulttextinterpreter','latex'); % allows you to use latex math
set(0,'defaultlinelinewidth',2); % line width is set to 2
set(0,'DefaultLineMarkerSize',10); % marker size is set to 10
set(0,'DefaultTextFontSize', 16); % Font size is set to 16
set(0,'DefaultAxesFontSize',16); % font size for the axes is set to 16
figure(2)
x_1=1:5000:T;
y_1=mean(regret_avg_UCB(:,x_1));
err=std(regret_avg_UCB(:,x_1));
errorbar(x_1,y_1,err);
hold on;
y_1=mean(regret_avg_TLP(:,x_1));
err=std(regret_avg_TLP(:,x_1));
errorbar(x_1,y_1,err);
hold on;
y_1=mean(regret_avg_UTLP(:,x_1));
err=std(regret_avg_UTLP(:,x_1));
errorbar(x_1,y_1,err);
grid on; % grid lines on the plot

%legend('HTS(good group)', 'HTS(Partial information)','TS','HTS(kmean)');
%legend('TS','MTS','UTS','MUTS','UCB');
legend('UCB','TLP','UTLP');
% ylabel('$Regret$ (Kbps)');
% 
% xlabel('$_$ (frames=sec)');
xlabel('Time Slot');
ylabel('Regret');
title('Error Bar');
x_1=1:1:T;
% 单因子方差分析
p_temp=[];
for i=1:T
% [p, tbl, stats] = anova1(data);
% fprintf('p 值 = %.4f\n', p);
data = [regret_avg_UCB(:,i)'; regret_avg_TLP(:,i)'; regret_avg_UTLP(:,i)'];
[p, h] = ranksum(data(1,:),data(2,:));
p_temp=[p_temp p];
end
p_temp1=[];
for i=1:T
data = [regret_avg_UCB(:,i)'; regret_avg_TLP(:,i)'; regret_avg_UTLP(:,i)'];
% [p, tbl, stats] = anova1(data);
% fprintf('p 值 = %.4f\n', p);
[p, h] = ranksum(data(1,:),data(3,:));
p_temp1=[p_temp1 p];
end
p_temp2=[];
for i=1:T
% [p, tbl, stats] = anova1(data);
% fprintf('p 值 = %.4f\n', p);
data = [regret_avg_UCB(:,i)'; regret_avg_TLP(:,i)'; regret_avg_UTLP(:,i)'];
[p, h] = ranksum(data(2,:),data(3,:));
p_temp2=[p_temp2 p];
end
set(0,'defaulttextinterpreter','latex'); % allows you to use latex math
set(0,'defaultlinelinewidth',2); % line width is set to 2
set(0,'DefaultLineMarkerSize',10); % marker size is set to 10
set(0,'DefaultTextFontSize', 16); % Font size is set to 16
set(0,'DefaultAxesFontSize',16); % font size for the axes is set to 16
figure(3)
%plot(x,0.98*packet_num_CUUCB(1:1000), '--c',x,packet_num_CUCB(1:1000),'-*',x,packet_num_CUUCB(1:1000),'m-.+',x,0.97*packet_num_CUCB_1(1:1000),'-bo',x,packet_num_CUCB_1(1:1000),'-ro','LineWidth',2,'MarkerIndices',1:100:length(x));
%plot(x,1.2*mean(regret_co3(:,1:1000)), '--c',x,mean(regret_co2(:,1:1000)),'-*',x,mean(regret_co33(:,1:1000)),'m-.+',x,mean(regret_co22(:,1:1000)),'-bo','LineWidth',2,'MarkerIndices',1:100:length(x));
%plot(X, Y1, '-bo', X, Y2, '{rs', X); % plotting three curves Y1, Y2 for the same X
%plot(x,regret,'--c',x,regret_2,'-*',x,regret_4,'m-.+','LineWidth',2,'MarkerIndices',1:10000:length(x));
%plot(x,regret,'--c',x,regret_2,'-*',x,regret_4,'m-.+','LineWidth',2,'MarkerIndices',1:10000:length(x));
%plot(x,mean(regret_avg_TS),'r--h',x,mean(regret_avg_MTS),'g-*',x,mean(regret_avg_UTS),'-bo',x,mean(regret_avg_MUTS),'c-s',x,mean(regret_avg_UCB),'m-.+','LineWidth',2,'MarkerIndices',1:5000:length(x));
plot(x_1,p_temp,'g-*',x_1,p_temp1,'c-s',x_1,p_temp2,'m-.+','LineWidth',2,'MarkerIndices',1:5000:length(x));
%plot(x,rate_HUUCB_1,'r--h',x,rate_HUUCB_2,'g-*','LineWidth',2,'MarkerIndices',1:5000:length(x));
grid on; % grid lines on the plot

%legend('HTS(good group)', 'HTS(Partial information)','TS','HTS(kmean)');
%legend('TS','MTS','UTS','MUTS','UCB');
legend('UCB vs TLP','UCB vs UTLP','TLP vs UTLP');
% ylabel('$Regret$ (Kbps)');
% 
% xlabel('$_$ (frames=sec)');
xlabel('Time Slot');
ylabel('P value');
title('Wilcoxon rank-sum test');
T=100000;
optimal_price=1;
iteration_num=50;
%percentage of UCB
p_UCB=[];
for i=1:T
p_UCB=[p_UCB length(find(a_avg_UCB(:,1:i)==optimal_price))/(iteration_num*i)];
end
%percentage of TLP
p_TLP=[];
for i=1:T
p_TLP=[p_TLP length(find(a_avg_TLP(:,1:i)==optimal_price))/(iteration_num*i)];
end
%percentage of UTSC
p_UTLP=[];
for i=1:T
p_UTLP=[p_UTLP length(find(a_avg_UTLP(:,1:i)==optimal_price))/(iteration_num*i)];
end
set(0,'defaulttextinterpreter','latex'); % allows you to use latex math
set(0,'defaultlinelinewidth',2); % line width is set to 2
set(0,'DefaultLineMarkerSize',10); % marker size is set to 10
set(0,'DefaultTextFontSize', 16); % Font size is set to 16
set(0,'DefaultAxesFontSize',16); % font size for the axes is set to 16
figure(4)
x=1:1:T;
%plot(x,0.98*packet_num_CUUCB(1:1000), '--c',x,packet_num_CUCB(1:1000),'-*',x,packet_num_CUUCB(1:1000),'m-.+',x,0.97*packet_num_CUCB_1(1:1000),'-bo',x,packet_num_CUCB_1(1:1000),'-ro','LineWidth',2,'MarkerIndices',1:100:length(x));
%plot(x,1.2*mean(regret_co3(:,1:1000)), '--c',x,mean(regret_co2(:,1:1000)),'-*',x,mean(regret_co33(:,1:1000)),'m-.+',x,mean(regret_co22(:,1:1000)),'-bo','LineWidth',2,'MarkerIndices',1:100:length(x));
%plot(X, Y1, '-bo', X, Y2, '{rs', X); % plotting three curves Y1, Y2 for the same X
%plot(x,regret,'--c',x,regret_2,'-*',x,regret_4,'m-.+','LineWidth',2,'MarkerIndices',1:10000:length(x));
%plot(x,regret,'--c',x,regret_2,'-*',x,regret_4,'m-.+','LineWidth',2,'MarkerIndices',1:10000:length(x));
%plot(x,mean(regret_avg_TS),'r--h',x,mean(regret_avg_MTS),'g-*',x,mean(regret_avg_UTS),'-bo',x,mean(regret_avg_MUTS),'c-s',x,mean(regret_avg_UCB),'m-.+','LineWidth',2,'MarkerIndices',1:5000:length(x));
plot(x,p_UCB,'g-*',x,p_TLP,'c-s',x,p_UTLP,'m-.+','LineWidth',2,'MarkerIndices',1:5000:length(x));
%plot(x,p_UCB(20:T),'r--h',x,p_TLP(20:T),'-bo',x,1.05*p_TLP(20:T),'m-.+','LineWidth',2,'MarkerIndices',1:5000:length(x));
%plot(x,p_TS(10:T),'r--h',x,p_MTS(10:T),'g-*',x,p_TLP(10:T),'-bo',x,p_MUTS(10:T),'c-s','LineWidth',2,'MarkerIndices',1:5000:length(x));
%plot(x,rate_HUUCB_1,'r--h',x,rate_HUUCB_2,'g-*','LineWidth',2,'MarkerIndices',1:5000:length(x));
grid on; % grid lines on the plot

%legend('HTS(good group)', 'HTS(Partial information)','TS','HTS(kmean)');
%legend('TS','MTS','UTS','MUTS','UCB');
legend('UCB','TLP','UTLP');
% ylabel('$Regret$ (Kbps)');
% 
% xlabel('$_$ (frames=sec)');

xlabel('Time Slot');
ylabel('Optimal Price Selected');
title('Percentage of The Optimal Arm Selected');


