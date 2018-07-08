function IMDB = setup_point(varargin)
% CNN_IMAGENET_SETUP_DATA  Initialize ImageNet ILSVRC CLS-LOC challenge data
%    This function creates an IMDB structure pointing to a local copy
%    of the ILSVRC CLS-LOC data.
%
%    The ILSVRC data ships in several TAR archives that can be
%    obtained from the ILSVRC challenge website. You will need:
%
%    ILSVRC2012_img_train.tar
%    ILSVRC2012_img_val.tar
%    ILSVRC2012_img_test.tar
%    ILSVRC2012_devkit.tar
%
%    Note that images in the CLS-LOC challenge are the same for the 2012, 2013,
%    and 2014 edition of ILSVRC, but that the development kit
%    is different. However, all devkit versions should work.
%
%    In order to use the ILSVRC data with these scripts, please
%    unpack it as follows. Create a root folder <DATA>, by default
%
%    data/imagenet12
%
%    (note that this can be a simlink). Use the 'dataDir' option to
%    specify a different path.
%
%    Within this folder, create the following hierarchy:
%
%    <DATA>/images/train/ : content of ILSVRC2012_img_train.tar
%    <DATA>/images/val/ : content of ILSVRC2012_img_val.tar
%    <DATA>/images/test/ : content of ILSVRC2012_img_test.tar
%    <DATA>/ILSVRC2012_devkit : content of ILSVRC2012_devkit.tar
%
%    In order to speedup training and testing, it may be a good idea
%    to preprocess the images to have a fixed size (e.g. 256 pixels
%    high) and/or to store the images in RAM disk (provided that
%    sufficient RAM is available). Reading images off disk with a
%    sufficient speed is crucial for fast training.

opts.dataDir = fullfile('data','pointing04') ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------
 imdb.imageDir = fullfile(opts.dataDir) ;

% -------------------------------------------------------------------------
%                                                           images
% -------------------------------------------------------------------------
files = struct2cell(dir( fullfile(imdb.imageDir)));
files(2:end,:) = [];
files(1:2) = [] ;

img_name = [];
img_label = [];
img_angle = [];
for i =1 : numel(files)
    names = struct2cell(dir( fullfile(imdb.imageDir,files{1,i},'*.jpg')));
    names(2:end,:) = [];    
    for j = 1:numel(names)
        labels(1,j) = str2double(names{1,j}(10:11))+1;
        angles{1,j} = names{1,j}(12:end-4);
    end
    img_name = [img_name fullfile(files{1,i},names)];
    img_label = [img_label labels];
    img_angle = [img_angle angles];
end

imdb.images.id = 1:numel(img_name) ;
imdb.images.name = img_name ;
imdb.images.set = ones(1, numel(img_name)) ;
imdb.images.label = img_label ;
imdb.meta.classes = img_angle(1:93) ;
% reshape(img_angle',93,[])

% -------------------------------------------------------------------------
%                                                         Train & Validation images
% -------------------------------------------------------------------------
indices = crossvalind('Kfold',img_label,5);
for i =1 :5
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