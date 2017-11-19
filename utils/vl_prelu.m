function [y,dzdw] = vl_nnprelu(x,w,dzdy)

x_size = [size(x,1), size(x,2), size(x,3), size(x,4)];
w_size = size(w) ;
w = reshape(w, [1 x_size(3) 1]) ;
x = reshape(x, [x_size(1)*x_size(2) x_size(3) x_size(4)]) ;


if nargin <= 2 || isempty(dzdy)
  y = max(x, single(0))+bsxfun(@times,w,min(x, single(0))) ;
  dzdw = [];
else
 dzdy = reshape(dzdy, size(x)) ; 
 dzdw = sum(sum(dzdy .* min(x, single(0)),1),3)  ;
 dzdw = reshape(dzdw, w_size) ;

 y = dzdy .* ((x > single(0))+ bsxfun(@times,w,x< single(0)));
end
y = reshape(y, x_size) ;

