function IMDB = setup_morph(varargin)
% Author: Bin-Bin Gao
% Email: gaobb@lamda.njuedu.cn
% modied 2015-09-07

% opts.autoDownload = false ;
opts.dataDir = fullfile('data','MORPH_Album2') ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

imdb.images.id = [] ;
imdb.images.set = uint8([]) ;
imdb.images.name = {} ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes =[];
imdb.imageDir = fullfile(opts.dataDir) ;

% Morph 
images_name = struct2cell( dir(fullfile(opts.dataDir,'*.*')));
images_name(2:end,:) = [];
images_name(1:2) =[];

img_age = [];
for num =1 : numel(images_name)
    name  = images_name{1,num};
    img_age(1,num) = max(str2double(name(end-5:end-4)),1);
    img_age(2,num) = 2;
end
% img_age(2,:) = img_age(2,:)+0.1*randn(1,num);
images =   fullfile(images_name);

imdb.images.name=images;
imdb.images.label = img_age;
imdb.images.id = 1:numel(imdb.images.name) ;

indices = crossvalind('Kfold',img_age(1,:),10);
for i =1 :10
sel_val = (indices ==i);
sel_train = ~sel_val;
imdb.images.set(sel_train') = 1;
imdb.images.set(sel_val') = 2;
IMDB{i} = imdb;
end

if opts.lite
    clear IMDB
    ok = {} ;
    for c = 23:32
        ok{end+1} = vl_colsubset(find(imdb.images.label(1,:) == c & imdb.images.set == 1), 50) ;
        ok{end+1} = vl_colsubset(find(imdb.images.label(1,:) == c & imdb.images.set == 2), 50) ;
        ok{end+1} = vl_colsubset(find(imdb.images.label(1,:) == c & imdb.images.set == 3), 50) ;
    end
    ok = cat(2, ok{:}) ;
    imdb.meta.classes = imdb.meta.classes ;
    imdb.images.id = imdb.images.id(ok) ;
    imdb.images.name = imdb.images.name(ok) ;
    imdb.images.set = imdb.images.set(ok) ;
    imdb.images.label = imdb.images.label(:,ok) ;
    IMDB{1,1} = imdb; 
end



