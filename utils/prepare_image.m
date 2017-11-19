function crops_data = prepare_image(images,IMAGE_DIM,mean)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
% d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
% mean_data = d.mean_data;

% IMAGE_DIM = 256;
CROPPED_DIM = 256;
mean_data = imresize(mean,[IMAGE_DIM IMAGE_DIM]);
im = imread(images{1});
if size(im,3) ==1
    im = cat(3,im,im.im);
end
% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
% im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
% im_data = permute(im_data, [2, 1, 3]);  % flip width and height
% im_data = single(im);  % convert from uint8 to single
% im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data

  w = size(im,2) ;
  h = size(im,1) ;
  factor = [IMAGE_DIM/h IMAGE_DIM/w];
  factor_max = [2048/h 2048/w];
  
 factor = min(max(factor), min(factor_max));
  
  if any(abs(factor - 1) > 0.0001)
    im = imresize(im, ...
                   'scale', factor, ...
                   'method', 'bilinear') ;
  end


im_data = single(im);
w = size(im,2) ;
h = size(im,1) ;
im_data = im_data - imresize(mean_data,[h,w]);  % subtract mean_data (already in W x H x C, BGR)
crops_data = im_data;
