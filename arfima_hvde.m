function [MLE_theta LL ut] = arfima_hvde(y,p,q,theta0)
%
% Gaussian MLE of the parameters of an arfima(p,q)
%
% p and q must belong to [0,1]
%

% Starting values for parameters
if nargin<4
theta0 = [.1; rand(p+q,1)./2];
end

% options = optimset('TolX',1e-12,'TolFun',1e-12,'Algorithm','interior-point'); %'sqp'
% options = optimset(options,'MaxFunEvals',10000,'Display','off');
options=optimset('Display','off','TolFun',1.0e-12,'TolX',1.0e-12);              % options for minimization
options  =  optimset(options , 'LargeScale'  , 'off');
options  =  optimset(options , 'GradObj'     , 'off');
options  =  optimset(options , 'Algorithm ', 'active-set');
LB = [-.5; -.99*ones(p+q,1)];
UB = [.5; .99*ones(p+q,1)];

par = fmincon(@(theta) log_likelihood_arfima(theta,y,p),theta0,[],[],[],[],LB,UB,[],options);

d=0; mm=0;
while (par(1)>.49999)||(par(1)<-.49999); %9999999
    d=d+par(1);
    delta = get_filter([d;par(2:end)],p);
    y_tilde = filter(delta,1,y);
    y_tilde(1,:)=[];
    par = fmincon(@(theta) log_likelihood_arfima(theta,y_tilde,p),par,[],[],[],[],LB,UB,[],options);
    mm=mm+1; if mm>5; d=d+par(1); break; end
end

d=d+par(1);

MLE_theta = [d;par(2:end)];
[LL, ut] = log_likelihood_arfima(MLE_theta,y,p);
LL=-LL;

end
%==========================================================================
% Auxiliary function computing the likelihood
%==========================================================================
function [LL ut] = log_likelihood_arfima(theta,y,p)

phi = theta(2:p+1);
theta_tilde = theta(p+2:end);

delta = get_filter(theta,p);
y_tilde = filter(delta,1,y);
ut = filter([1 -phi],[1 +theta_tilde],y_tilde); ut(1)=[];
% lambda = get_filter(theta,p);
% 
% initial = repmat(y(1),length(lambda),1);
% reg = [initial;y];
% ut  = filter(lambda,1,reg);
% ut  = ut(length(lambda)+1:end);

s2_u = cov(ut);
LL=s2_u;
% LL = -sum( -0.5*(log(2*pi)+log(s2_u)+ ut.^2 ));
% 
% if isnan(LL)
%     LL=1e6;
% end

end
%==========================================================================
% Auxiliary function computing the filter
%==========================================================================
function delta_k = get_filter(theta,p)%#ok
T=evalin('caller','length(y)');
filter_lag = ceil(min(T/4,2*sqrt(T)));

d   = theta(1);
% phi = theta(2:p+1);
% theta_tilde = -theta(p+2:end);
% 
% if isempty(theta_tilde), theta_tilde=0; end
% if isempty(phi), phi=0; end
% 
% lambda = zeros(filter_lag,1);
% Psi_k = zeros(filter_lag,1);
delta_k = [1 cumprod(((0:1:filter_lag-1)-d)./(1:1:filter_lag))];
% Psi_k(1) = 1;
% lambda(1) = 1;
% for i=2:filter_lag
%     Psi_k(i) = theta_tilde*Psi_k(i-1) + delta_k(i);
%     lambda(i)= Psi_k(i) - phi*Psi_k(i-1);
% end
end
