%% function testing 
seed = 1357; 
rng(seed,'twister');
% process state equation 
% x = Fx(k-1) + Gu(k-1) 
% initialize x(0) ~ N(m0,Pi0) 
% measurement equation
% r = Cx(k) + w(k)
N = 1000;

F = [1 0; 1 0.5]; % state transition matrix
C = [1 0; 0.5 1]; % observation matrix
G = [1 0 1; 1 0 0]%0.1*eye(2); % process noise covariance
R = 1*eye(2); % measurement noise covariance
sigmaq = 1;
m0 = [20; 30]; % initial state estimate
P = 0.5*eye(2); % initial covariance matrix

[x,r] = Kalman_Signal_Model(F,G,C,R,sigmaq,m0,P,N);

subplot(2,1,1)
plot(r(1,:),'.-')

grid on 

subplot(2,1,2)
plot(r(2,:),'.')

grid on 

%% function
function [x,r] = Kalman_Signal_Model(F,G,C,R,sigmaq,m0,P,N)

% initialize x0
     
     [~,Nx] = size(F);
     [~,Nu] = size(G);

     [Nr,~] = size(C);

     if P == 0 
         x0 = m0;
     else
         sigma = chol(P);
         x0 = repmat(m0,1,N) + sqrtm(sigma)*randn(Nx,N);
     end
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
            x(:,i) = F*x0(:,i);
        else
            x(:,i) = F*x(:,i-1) + G*u(:,i-1); 
        end
    end

    r = C*x + w; 
    % update observations

end
