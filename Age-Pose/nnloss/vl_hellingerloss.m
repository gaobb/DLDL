function Y = vl_hellingerloss(x, x0,dzdy)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2015-09-07

% sz = [size(x,1) size(x,2) size(x,3) size(x,4)] ;
% n = sz(4);
% index from 0
p(1,1,:,:) = x0;

q = max(min(x,1-10^-15),10^-15);
pq = sqrt(p) - sqrt(q); 

if nargin <= 2
    t = pq.*pq;
    Y =  sum(t(:)) ;
else
    Y = -pq./sqrt(q).*dzdy; 
end