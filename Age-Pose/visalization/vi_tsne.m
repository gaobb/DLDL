clc
clear

root_path = 'model_v2/aflw_det-izfnet-simplenn/klloss-3721/fold1';
% imdbstruct = load('model_v2/aflw_det-izfnet-simplenn/new_imdb.mat') ;
imdbstruct = load('model_v2/aflw_det-izfnet-simplenn/imdb.mat') ;
% load(fullfile(root_path,'new_fc8_feat.mat'));
load(fullfile(root_path,'mappedX.mat'))
imdb = imdbstruct.IMDB{1,1};
imdb.imageDir = imdb.images.crop_imageDir;
imdb.images.name = imdb.images.crop_name;


id = find(imdb.images.set == 2);
% val_labels =  round(imdb.images.label(1:2,id));
val_labels =  imdb.images.label(1:2,id);
val_labels(1,:) = -1*val_labels(1,:);
val_labels(1,:) = val_labels(2,:);

root_path = 'model_v2/ChaLearn-izfnet-simplenn/klloss/fold1';
% root_path = 'model_v2/ChaLearn-izfnet-simplenn/klloss/fold1';
imdbstruct = load('model_v2/ChaLearn-izfnet-simplenn/imdb.mat') ;
imdb = imdbstruct.IMDB{1,1};
id = find(imdb.images.set == 2);
val_labels =  round(imdb.images.label(1,id));
 load(fullfile(root_path,'mappedX.mat'))


% train_labels = imdb.images.label(1:2,val);
%  train_labels = imdb.images.label(1:2,val);
root_path = 'model_v2/MORPH_Album2-izfnet-simplenn/klloss/fold1';
imdbstruct = load('model_v2/MORPH_Album2-izfnet-simplenn/imdb.mat') ;
imdb = imdbstruct.IMDB{1,1};
imdb.imageDir = '/home/gaobb/mywork/SV3/image_data/age_faces/MORPH_Album2_align';
id = find(imdb.images.set == 2);
val_labels =  round(imdb.images.label(1,id));
 load(fullfile(root_path,'mappedX.mat'))

%% load embedding

% load('imagenet_val_embed.mat'); % load x (the embedding 2d locations from tsne)
x = mappedX;
x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));

gscatter(x(:,2), 1-x(:,1), val_labels(1,:),[],'.',8,'off');
% set(gca, 'CLim', [0 90])
% colorbar
% gscatter(x,y,g,clr,sym,siz,doleg,xnam,ynam)
% set(gca,'ytick',[]);
% set(gca,'xtick',[]);
box off
axis off
% print(gcf,'-depsc',['figure/','morph_age.eps'])
%print(gcf,'-depsc',['figure/','aflw_yaw.eps'])
% print(gcf,'-depsc',['figure/','aflw_pitch.eps'])
print(gcf,'-depsc',['figure/','chalearn_age.eps'])


id = find(imdb.images.set==2);
% image_paths = fullfile(imdb.images.crop_imageDir,imdb.images.crop_name(id));

image_paths = fullfile(imdb.imageDir,imdb.images.name(id));
fs = image_paths;
N = length(fs);
fs'
%% create an embedding image

S = 2000; % size of full embedding image
G = 255.*ones(S, S, 3, 'uint8');
s = 48; % size of every single image

Ntake = N;
for i=1:Ntake
    
    if mod(i, 100)==0
        fprintf('%d/%d...\n', i, Ntake);
    end
    
    % location
    a = ceil(x(i, 1) * (S-s)+1);
    b = ceil(x(i, 2) * (S-s)+1);
    a = a-mod(a-1,s)+1;
    b = b-mod(b-1,s)+1;
    if G(a,b,1) ~= 255
        continue % spot already filled
    end
    
    I = imread(fullfile(fs{i}));
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);
    
    G(a:a+s-1, b:b+s-1, :) = I;
    
end
% imwrite(G, 'chalearn_dldl_48.jpg', 'jpg');

imwrite(G, 'figure/chalearn_dldl_48.jpg', 'jpg');
imwrite(G, 'figure/morph_dldl_48.jpg', 'jpg');
imwrite(G, 'figure/alfw_dldl_48.jpg', 'jpg');

