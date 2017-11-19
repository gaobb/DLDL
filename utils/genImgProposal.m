function im_data = genImgProposal(im, edge_model, mean_data)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
IMAGE_DIM = 256;
%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect


if size(im,3) == 1
    im = cat(3, im, im, im) ;
end
[h, w, c] = size(im);
% detect Edge Box bounding box proposals (see edgeBoxes.m)
bbs = edgeBoxes(im, edge_model, opts);  %% bbs = [x y w h]

% fiflter bbox (areas<900 pixels and height/width >4)
areas = bbs(:,3).*bbs(:,4);
hwr = bbs(:,4)./bbs(:,3);
whr = bbs(:,3)./bbs(:,4);

ind = areas<900 | hwr >4 | whr >4;
%fprintf('filter %d bboxes\n',sum(ind));
bbs(ind,:) = [];
%% compute IOU
% [x y w h] to [x1 y1 x2 y2]
bbs_num = size(bbs,1);
bbs(:,3:4) = bbs(:,1:2) + bbs(:,3:4);
W = zeros(bbs_num,bbs_num);
for i=1:size(bbs,1)
    W(i,1:i-1) = IOUs(bbs(i,:),bbs(1:i-1,:));
end
W = W + W' + eye(bbs_num, bbs_num);
%     for i= 1:100
%         pairs = [datasample(1:bbs_num,2)];
%         imshowboxes(im,bbs(pairs,:));
%         title(W(pairs(1),pairs(2)))
%         pause
%     end
%% normalized cut
nbCluster = 15;
%     tic;
[NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(W,nbCluster);
%     disp(['The computation took ' num2str(toc) ' seconds']);
sel_bbs = [];
top_k = 30;
for i =1:nbCluster
    ind = find(NcutDiscrete(:,i)==1);
    sel_ind = ind(1:min(top_k,numel(ind)));
    sel_bbs = [sel_bbs; bbs(sel_ind,:)];
%     imshowboxes(im,bbs(sel_ind(1:min(30,numel(sel_ind))),:));
%     pause
end

sel_bbs_num = min(size(sel_bbs, 1), 450);
patches = single(zeros(IMAGE_DIM,IMAGE_DIM,3, sel_bbs_num));
for p = 1:sel_bbs_num
    patch  = im(sel_bbs(p, 2):sel_bbs(p, 4), sel_bbs(p, 1):sel_bbs(p,3),:);
%     imshow(patch)
%     pause
    patches(:,:,:,p)  = imresize(patch, [IMAGE_DIM,IMAGE_DIM], 'bilinear');
end
% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
% im_data = patches(:, :, [3, 2, 1], :);  % permute channels from RGB to BGR
% im_data = permute(im_data, [2, 1, 3, 4]);  % flip width and height

im_data = single(patches);  % convert from uint8 to single
im_data = bsxfun(@minus, im_data ,imresize(mean_data, [IMAGE_DIM,IMAGE_DIM], 'bilinear'));  % subtract mean_data (already in W x H x C, BGR)
