function IMDB = setup_bjut(varargin)
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

opts.dataDir = fullfile('data','bjut-3d') ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------
 imdb.imageDir = fullfile(opts.dataDir) ;

% -------------------------------------------------------------------------
%                                                           images
% -------------------------------------------------------------------------
names = struct2cell(dir( fullfile(imdb.imageDir,'*.bmp')));
names(2:end,:) = [];
id = [];
labels = [];
for i =1 : numel(names)
    id(1,i) = str2double(names{1,i}(1:find(names{1,i}=='_')-1));
    labels(1,i) = str2double(names{1,i}(find(names{1,i}=='_')+1:end-4));
end

imdb.images.id = 1:numel(names) ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;

% -------------------------------------------------------------------------
%                                                         Train & Validation images
% -------------------------------------------------------------------------
indices = crossvalind('Kfold',labels,5);
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