function y = vl_nnklloss(x,ld,dzdy)
%VL_NNSOFTMAXLOSS CNN combined softmax and logistic loss.
%   **Deprecated: use `vl_nnloss` instead**
%
%   Y = VL_NNSOFTMAX(X, C) applies the softmax operator followed by
%   the kl loss the data X. X has dimension H x W x D x N,
%   packing N arrays of W x H D-dimensional vectors.
%
%   C contains the class labels, which should be integers in the range
%   1 to D. C can be an array with either N elements or with dimensions
%   H x W x 1 x N dimensions. In the fist case, a given class label is
%   applied at all spatial locations; in the second case, different
%   class labels can be specified for different locations.
%
%   DZDX = VL_NNSOFTMAXLOSS(X, C, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% work around a bug in MATLAB, where native cast() would slow
% progressively

% c = l(1,:);
% sigma = l(2,:);
% ld = genLd(c,sigma);
y_(1,1,:,:) = ld;

sz_c = size(y_);
sz_x = size(x);
assert(isequal(sz_c, sz_x)) ;

% compute softmaxloss
xmax = max(x,[],3) ;
ex = exp(bsxfun(@minus, x, xmax)) ;

if nargin <= 2
  t = y_.*(bsxfun(@minus, bsxfun(@minus, x, xmax), log(sum(ex,3))));
  y = -sum(t(:)) ;
else
  y = bsxfun(@rdivide, ex, sum(ex,3)) ;
  y = y - y_;
  y = bsxfun(@times, y, dzdy) ;
end
end