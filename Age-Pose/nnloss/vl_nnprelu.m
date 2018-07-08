function [y,dzdw] = vl_nnprelu(x,w,dzdy)
% VL_NNRELU  CNN rectified linear unit
%   Y = VL_NNRELU(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = VL_NNRELU(X, DZDY) computes the network derivative DZDX
%   with respect to the input X given the derivative DZDY with respect
%   to the output Y. DZDX has the same dimension as X.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = VL_NNRELU(X), then
%   VL_NNRELU(X,DZDY) gives the same result as VL_NNRELU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
x_size = [size(x,1), size(x,2), size(x,3), size(x,4)];
w_size = size(w) ;
w = reshape(w, [1 x_size(3) 1]) ;
x = reshape(x, [x_size(1)*x_size(2) x_size(3) x_size(4)]) ;

% temp_w = repmat(w,sz(1),sz(2),1,sz(4));

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

