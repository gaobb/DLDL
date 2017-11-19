function IMDB = setup_aflw(varargin)

opts.dataDir = './data' ;
opts.dataset = 'aflw';
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------
imdb.imageDir = fullfile(opts.dataDir,'aflw_det');
data_label = load(fullfile(opts.dataDir,'aflw_det','aflw.mat'));
data_label.image.pose(1,24396-11:24396-1);
data_label.image.name(1,24396-11:24396-1);

for n = 1:24396-12
    poses(:,n) = data_label.image.pose{1,n};
    names{:,n} = data_label.image.name{1,n};
end
poses(:,n+1) = data_label.image.pose{1,24396};
names{:,n+1} = data_label.image.name{1,24396};
id = intersect(intersect(find(poses(1,:)>-pi/2 & poses(1,:)<pi/2),find(poses(2,:)>-pi/2 & poses(2,:)<pi/2)),find(poses(3,:)>-pi/2 & poses(3,:)<pi/2));
% -------------------------------------------------------------------------
%                                                           images
% -------------------------------------------------------------------------
imdb.images.id = 1:numel(id) ;
imdb.images.name = names(1,id) ;
imdb.images.set = ones(1,numel(id)) ;
imdb.images.label = poses(:,id)*180/pi ;

% -------------------------------------------------------------------------
%                                                         Train & Validation images
% -------------------------------------------------------------------------
indices = crossvalind('Kfold',imdb.images.label(2,:),3);
for i =1:3
sel_val = (indices ==i);
sel_train = ~sel_val;
imdb.images.set(sel_train') = 1;
imdb.images.set(sel_val') = 2;
IMDB{i} = imdb;
end
% -------------------------------------------------------------------------
%                                                            Postprocessing
% -------------------------------------------------------------------------

% sort categories by WNID (to be compatible with other implementations)
% [imdb.classes.name,perm] = sort(imdb.classes.name) ;
% imdb.classes.description = imdb.classes.description(perm) ;
% relabel(perm) = 1:numel(imdb.classes.name) ;
% ok = imdb.images.label >  0 ;
% imdb.images.label(ok) = relabel(imdb.images.label(ok)) ;

if opts.lite
  % pick a small number of images for the first 10 classes
  % this cannot be done for test as we do not have test labels
  clear keep ;
  clear IMDB;
  for i=1:10
    sel = find(imdb.images.label == i) ;
    train = sel(imdb.images.set(sel) == 1) ;
    val = sel(imdb.images.set(sel) == 2) ;
    train = train(1:20) ;
    val = val(1:10) ;
    keep{i} = [train val] ;
  end
  keep = sort(cat(2, keep{:}) );
  imdb.images.id = imdb.images.id(keep) ;
  imdb.images.name = imdb.images.name(keep) ;
  imdb.images.set = imdb.images.set(keep) ;
  imdb.images.label = imdb.images.label(keep) ;
  IMDB{1,1} = imdb; 
end


IMDB{1,1}.images.name = imdb.images.name;
IMDB{1,2}.images.name = imdb.images.name;
IMDB{1,3}.images.name = imdb.images.name;

IMDB{1,1}.crop_imageDir = '/mnt/data3/gaobb/image_data/image_faces/head_pose/aflw_det';
IMDB{1,2}.crop_imageDir = '/mnt/data3/gaobb/image_data/image_faces/head_pose/aflw_det';
IMDB{1,3}.crop_imageDir = '/mnt/data3/gaobb/image_data/image_faces/head_pose/aflw_det';