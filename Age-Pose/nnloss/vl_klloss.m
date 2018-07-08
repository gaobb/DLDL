function Y = vl_klloss(x, x0,dzdy)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2015-09-07

% sz = [size(x,1) size(x,2) size(x,3) size(x,4)] ;
% n = sz(4);
% index from 0
c(1,1,:,:) = x0;

x = max(min(x,1-10^-15),10^-15);

if nargin <= 2
     t =  c.* log(x); % KL loss
     Y =  -sum(t(:)) ;
else
     Y = -1./x.*(dzdy*(c(1,1,:,:))); %
end
