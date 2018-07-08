function Y = vl_epsilonloss(x, x0,dzdy)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2015-09-07

% sz = [size(x,1) size(x,2) size(x,3) size(x,4)] ;
% n = sz(4);
% index from 0
c(1,1,:,:) = x0;
epsilon = 2/180;%2/180
% x = max(min(x,1-10^-15),10^-15);
t =  abs(c-x); % KL
if nargin <= 2
     t(t<epsilon) = 0;
%     t =  (c(1,1,:,:) .* log(x) + x.* log(c))./2; %symmetric KL
    Y = sum(t(:)) ;
else 
    Y = sign(x-c).*dzdy; %
    Y(t<epsilon) = 0;
%  Y = -(1./x.*(dzdy*c(1,1,:,:))+ dzdy*log(c))./2;
end
