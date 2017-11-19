function imo = getPFDLDLbatch(images, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [227, 227] ;
opts.border = [29, 29] ;
opts.keepAspect = true ;
opts.numAugments = 1 ;
opts.flip = false ;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 30 ;
opts.prefetch = false ;
opts.edge_model = [];
opts = vl_argparse(opts, varargin);

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = numel(images) >= 1 && ischar(images{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

if prefetch
  vl_imreadjpeg(images, 'numThreads', opts.numThreads, 'prefetch') ;
  imo = [] ;
  return ;
end
if fetch
  im = vl_imreadjpeg(images,'numThreads', opts.numThreads) ;
else
  im = images ;
end

if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
  opts.averageImage = zeros(1,1,3) ;
end
if numel(opts.averageImage) == 3
  opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end
%% set up opts for edgeBoxes (see edgeBoxes.m)
eopts = edgeBoxes;
eopts.alpha = .65;     % step size of sliding window search
eopts.beta  = .75;     % nms threshold for object proposals
eopts.minScore = .01;  % min score of boxes to detect
eopts.maxBoxes = 1e4;  % max number of boxes to detect
tic
for n = 1:numel(im)
    imt = im{n};
    if opts.flip & rand > 0.5
       imt = imt(:,end:-1:1,:);
    end
%     imshow(uint8(im{n}))
    bbs=edgeBoxes(uint8(imt),opts.edge_model,eopts);   %% bbs = [x y w h]
%     fprintf('detect %d bboxes\n',size(bbs,1));
    
    %% fiflter bbox (areas<900 pixels and height/width >4)
    maxAspectRatio = 4; %- [3] max aspect ratio of boxes
    minBoxArea = 900;   % - [1000] minimum area of boxes
    
    areas = bbs(:,3).*bbs(:,4);
    hwr = bbs(:,4)./bbs(:,3);
    
    ind = areas<minBoxArea | hwr >maxAspectRatio | 1./hwr >maxAspectRatio;
%     fprintf('filter %d bboxes\n',sum(ind));
    bbs(ind,:) = [];
    bbs_num = size(bbs,1);
    %% compute IOU
    bbs(:,3:4) = bbs(:,1:2) + bbs(:,3:4) - 1;
    W = zeros(bbs_num,bbs_num);
    for i=1:bbs_num
        W(i,1:i-1) = IOUs(bbs(i,:),bbs(1:i-1,:));
    end
    W = W+W';
    W = W+eye(bbs_num, bbs_num);
    %% normalized cut
    nbCluster = 15;
%     tic;
    [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(W,nbCluster);
%     disp(['The computation took ' num2str(toc) ' seconds']);
%     hind = zeros(1, nbCluster);
%     for i =1:nbCluster
%         hinds = find(NcutDiscrete(:,i)==1);
%         if isempty(hinds)
%             hind(1,i) = datasample(1:bbs_num,1);
%         else
%             hind(1,i) = hinds(1);
%         end
%     end
%     hpatch_bbs =  bbs(hind,:);
%     imshowboxes(uint8(imt),bbs);
    [h,w,c] = size(imt);
    imt = bsxfun(@minus, imt, imresize(opts.averageImage, [h,w])) ;
    patches = single(zeros(256,256,3,nbCluster));
    for i = 1:nbCluster
        hinds = find(NcutDiscrete(:,i)==1);
        if isempty(hinds)
            hind = datasample(1:bbs_num,1);
        else
            hind = hinds(1);
        end
        patch = imt(bbs(hind,2):bbs(hind,4),bbs(hind,1):bbs(hind,3),:);
        patches(:,:,:,i) = imresize(patch, [256,256], 'bilinear');
%         imshow(uint8(patches(:,:,:,i)))
%         pause;
    end
    start_p = 1 + (n-1)*nbCluster;
    end_p = start_p + nbCluster -1;
    imo(:,:,:,start_p:end_p) = patches;
end





% imo = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
%             numel(images)*opts.numAugments, 'single') ;
% 
% si = 1 ;
% for i=1:numel(images)
% 
%   % acquire image
%   if isempty(im{i})
%     imt = imread(images{i}) ;
%     imt = single(imt) ; % faster than im2single (and multiplies by 255)
%   else
%     imt = im{i} ;
%   end
%   if size(imt,3) == 1
%     imt = cat(3, imt, imt, imt) ;
%   end
% 
%   % resize
%   w = size(imt,2) ;
%   h = size(imt,1) ;
%   factor = [(opts.imageSize(1)+opts.border(1))/h ...
%             (opts.imageSize(2)+opts.border(2))/w];
% 
%   if opts.keepAspect
%     factor = max(factor) ;
%   end
%   if any(abs(factor - 1) > 0.0001)
%     imt = imresize(imt, ...
%                    'scale', factor, ...
%                    'method', opts.interpolation) ;
%   end
% 
%   % crop & flip
%   w = size(imt,2) ;
%   h = size(imt,1) ;
%   for ai = 1:opts.numAugments
%     switch opts.transformation
%       case 'stretch'
%         sz = round(min(opts.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [h;w])) ;
%         dx = randi(w - sz(2) + 1, 1) ;
%         dy = randi(h - sz(1) + 1, 1) ;
%         flip = rand > 0.5 ;
%       otherwise
%         tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
%         sz = opts.imageSize(1:2) ;
%         dx = floor((w - sz(2)) * tf(2)) + 1 ;
%         dy = floor((h - sz(1)) * tf(1)) + 1 ;
%         flip = tf(3) ;
%     end
%     sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
%     sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
%     if flip, sx = fliplr(sx) ; end
% 
%     if ~isempty(opts.averageImage)
%       offset = opts.averageImage ;
%       if ~isempty(opts.rgbVariance)
%         offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3)) ;
%       end
%       imo(:,:,:,si) = bsxfun(@minus, imt(sy,sx,:), offset) ;
%     else
%       imo(:,:,:,si) = imt(sy,sx,:) ;
%     end
%     si = si + 1 ;
%   end
% end
