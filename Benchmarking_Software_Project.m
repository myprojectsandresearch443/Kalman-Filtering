%% function testing with bench marking example
%% 
% To run the function make sure rows of F and G matrix are same. 
% To run the function please define R as a matrix. 


close all 
clc
clear

seed = 16*2+64+256; 
rng(seed,'twister');


m0 = [1000; -50];
sigmaq = 40;
sigmar = 100;
T = 0.1; 

F = [1 T; 0 1];
G = [(T^2)/2; T];
C = [1 0];
P = 1000*[1 0; 0 1];
R = 100;
N = 150;

[x,r] = Kalman_Signal_Model(F,G,[1,0],sigmar,sigmaq,m0,P,N); 
[x1,r1] = Kalman_Signal_Model_NoNoise(F,G,[1,0],sigmar,sigmaq,m0,P,N);

[x_hat,P_hat,K] = Kalman_filter([0;0],P,F,G,C,R,sigmaq,N,r);

%
figure
t = linspace(0,10,N);

subplot(2,1,1)
plot(t,r(1,:),'.-')
hold on 
plot(t,x1(1,:));
plot(t,x_hat(1,:),'.-');
title('Range')
xlabel('t(s)')
ylabel('r')
legend('Noisy','True','Estimated')

grid on 
hold off

subplot(2,1,2)
hold on
plot(t,x(2,:),'.')

plot(t,x_hat(2,:))
title('Velocity')
xlabel('t(s)')
ylabel('v')
legend('True','Estimated')

grid on 
hold off
%%
figure 
subplot(1,2,1)

plot(t,K(1,:),'.-')
title('Kalman Gain - position')
xlabel('t(s)')
ylabel('Kp')
ylim([0 1.5])
grid on
subplot(1,2,2)
plot(t,K(2,:),'.-')
title('Kalman Gain - velocity')
xlabel('t(s)')
ylabel('Kv')
ylim([0 1.5])
grid on

%%
figure 
subplot(1,2,1)

semilogy(t,reshape(P_hat(1,1,:),[],1),'.-')
%hold on
%semilogy(t,mean(abs(x_hat(1,:)-r(1,:)).^2,1),'.-')
ylim([1e0 1e4])
grid on
title('MSE- position')
xlabel('t(s)')
ylabel('m^2')

subplot(1,2,2)
semilogy(t,reshape(P_hat(2,2,:),[],1),'.-')
%hold on
%semilogy(t,mean(abs(x_hat(2,:)-x(2,:)).^2,1),'.-')
ylim([1e0 1e4])
grid on
title('MSE - velocity')
xlabel('t(s)')
ylabel('m^2/s^2')
%% function
function [x,r] = Kalman_Signal_Model(F,G,C,R,sigmaq,m0,P,N)

% initialize x0
     
     [~,Nx] = size(F);
     [~,Nu] = size(G);

     [Nr,~] = size(C);

     x0 = m0;
%      if P == 0 
%          x0 = m0;
%      else
%          sigma = chol(P);
%          x0 = repmat(m0,1,N) + sqrtm(sigma)*randn(Nx,N);
%      end
     if Nu>1
         u = sqrtm((sigmaq*eye(Nu)))*randn(Nu,N);
     else
         u = sqrt(sigmaq)*randn(Nu,N);
     end

     if Nr > 1
        w = sqrtm(chol(R))*randn(Nr,N);
     else
        w = sqrt(R(Nr,Nr))*randn(Nr,N);
     end
    % update state loop 
    x = zeros(Nx,N);
    for i = 1:N
        if i == 1 
            x(:,i) = x0;
        else
            x(:,i) = F*x(:,i-1) + G*u(:,i-1); 
        end
    end

    r = C*x + w; 
    % update observations

end


function [x_hat,P_hat,K] = Kalman_filter(m0,P,F,G,C,R,sigmaq,N,r)
    x_hat_0 = m0;
    P0 = P;
    
    [~,Nx] = size(F);
    [~,Nu] = size(G);
    [Nr,~] = size(C);
    Q = (sigmaq*eye(Nu));
    x_hat_gk = zeros(Nx,N);
    for i = 1:N
        if i == 1 
            x_hat(:,i) = x_hat_0;%x_hat_gk(:,i) + K(:,:,i)*(r(:,i) - C*x_hat_gk(:,i));
            P_hat(:,:,i) = P0;%P_gk(:,:,i) - K(:,:,i)*C*P_gk(:,:,i);
             
        else
            x_hat_gk(:,i) = F*x_hat(:,i-1);
            P_gk(:,:,i) = F*P_hat(:,:,i-1)*F' + G*Q*G';
            K(:,:,i) = P_gk(:,:,i)*C'*inv(C*P_gk(:,:,i)*C' + R);
            x_hat(:,i) = x_hat_gk(:,i) + K(:,:,i)*(r(:,i) - C*x_hat_gk(:,i));
            P_hat(:,:,i) = P_gk(:,:,i) - K(:,:,i)*C*P_gk(:,:,i);
        end
    end

end


function [x,r] = Kalman_Signal_Model_NoNoise(F,G,C,R,sigmaq,m0,P,N)

% initialize x0
     
     [~,Nx] = size(F);
     [~,Nu] = size(G);

     [Nr,~] = size(C);
        
     x0 = m0;
%      if P == 0 
%          x0 = m0;
%      else
%          sigma = chol(P);
%          x0 = repmat(m0,1,N) + sqrtm(sigma)*randn(Nx,N);
%      end
     if Nu>1
         u = sqrtm(chol(sigmaq*eye(Nu)))*randn(Nu,N);
     else
         u = sqrt(sigmaq*0.5)*randn(Nu,N);
     end

     if Nr > 1
        w = sqrtm(chol(R))*randn(Nr,N);
     else
        w = sqrt(R(Nr,Nr))*randn(Nr,N);
     end
    % update state loop 
    x = zeros(Nx,N);
    for i = 1:N
        if i == 1 
            x(:,i) = x0;
        else
            x(:,i) = F*x(:,i-1) + G*u(:,i-1); 
        end
    end
    
    r = C*x;% + w; 
    % update observations

end
