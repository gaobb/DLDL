function Y = vl_klloss(x, x0,dzdy)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2015-09-07
% not cluding softmax layer
c(1,1,:,:) = x0;
x = max(min(x,1-10^-15),10^-15);

if nargin <= 2
     t =  c.* log((x));
     Y =  -sum(t(:)) ;
else
     Y = -1./x.*(dzdy*(c));
end
