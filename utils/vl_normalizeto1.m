function y = vl_normalizeto1(x, dzdy)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2016-12-22
if nargin <= 1
    % normalize
    y = bsxfun(@times, x, 1./sum(x,4));
else
    yn = bsxfun(@times, x, 1./sum(x,4));
    y = bsxfun(@times, 1./sum(x,4) , (1- yn).* dzdy);
end