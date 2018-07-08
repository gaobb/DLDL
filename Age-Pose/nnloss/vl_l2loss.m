function Y = vl_l2loss(x, x0,dzdy)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2015-09-07

c(1,1,:,:) = x0;

if nargin <= 2
    t =  0.5*(c-x).^2; 
    Y =  sum(t(:)) ;
else
    Y = (x-c).*dzdy; 
end
