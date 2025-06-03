seed = 16*2+64+256; 
rng(seed,'twister');
% process state equation 
% x = Fx(k-1) + Gu(k-1) 
% initialize x(0) ~ N(m0,Pi0) 
% measurement equation
% r = Cx(k) + w(k)
N = 1000;
% to test 
F = [1 0.1; 0 0.9]; % state transition matrix
C = [1 0; 0 1]; % observation matrix
G = [1 0 1; 0 0 1]%0.1*eye(2); % process noise covariance
R = 1*eye(2); % measurement noise covariance
sigmaq = 1;
m0 = [0; 0]; % initial state estimate
P = eye(2); % initial covariance matrix

[x,r] = Kalman_Signal_Model(F,G,C,R,sigmaq,m0,P,N);

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
         x0 = repmat(m0,1,N) + sigma*randn(Nx,N);
     end
     u = sqrtm(chol(sigmaq*eye(Nu)))*randn(Nu,N);
     w = sqrtm(chol(R))*randn(Nx,N);
    % update state loop 
    for i = 1:N-1
        if i == 1 
            x(:,i+1) = F*x0(:,i) + G*u(:,i);
        else
            x(:,i+1) = F*x(:,i-1) + G*u(:,i); 
        end
    end

    r = C*x + w; 
    % update observations

end