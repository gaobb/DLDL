function y = vl_hcp(x, dzdy)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2016-12-22
clusters = 15;
xt = permute(x, [4,1,2,3]);
if nargin <= 1
    yt = vl_nnpool(xt, [clusters, 1], ...
                   'pad', 0, 'stride', [clusters, 1], ...
                   'method', 'Max') ;
    y = permute(yt, [2,3,4,1]);
else
    dzdyt = permute(dzdy, [4,1,2,3]);
    yt = vl_nnpool(xt, [clusters, 1], dzdyt, ...
        'pad', 0, 'stride', [clusters, 1], ...
        'method', 'Max');
    y = permute(yt, [2,3,4,1]);
end