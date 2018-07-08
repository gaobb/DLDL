clear
clc

% addpath(genpath('drtoolbox'));

root_path = 'model_v2/aflw_det-izfnet-simplenn/klloss-3721/fold1';
root_path = 'model_v2/aflw_det-izfnet-simplenn/bfgs_ldl';

imdbstruct = load('model_v2/aflw_det-izfnet-simplenn/imdb.mat') ;
% load(fullfile(root_path,'new_fc8_feat.mat'));
load(fullfile(root_path,'mappedX.mat'))
imdb = imdbstruct.IMDB{1,1};
id = find(imdb.images.set == 2);
val_labels =  round(imdb.images.label(1:2,id));
val_labels(1,:) = -1*val_labels(1,:);

root_path = 'model_v2/ChaLearn-izfnet-simplenn/klloss/fold1';
% root_path = 'model_v2/ChaLearn-izfnet-simplenn/bfgs_ldl';
load(fullfile(root_path,'mappedX.mat'))

imdbstruct = load('model_v2/ChaLearn-izfnet-simplenn/imdb.mat') ;
imdb = imdbstruct.IMDB{1,1};
id = find(imdb.images.set == 2);
val_labels =  round(imdb.images.label(1,id));
%  load(fullfile(root_path,'mappedX.mat'))


% train_labels = imdb.images.label(1:2,val);
%  train_labels = imdb.images.label(1:2,val);
root_path = 'model_v2/MORPH_Album2-izfnet-simplenn/klloss/fold1';
root_path = 'model_v2/MORPH_Album2-izfnet-simplenn/bfgs_ldl';

load(fullfile(root_path,'mappedX.mat'))
imdbstruct = load('model_v2/MORPH_Album2-izfnet-simplenn/imdb.mat') ;
imdb = imdbstruct.IMDB{1,1};
id = find(imdb.images.set == 2);
val_labels =  round(imdb.images.label(1,id));
val_names = imdb.images.name(1,id);
% load(fullfile(root_path,'mappedX.mat'))

 
 
root_path = 'model_v2/MORPH_Album2-izfnet-simplenn/bfgs_ldl';
load(fullfile(root_path,'val_bif_feat.mat'));
cnn_feats = val_bif;
val_labels = val_label;
% load('model_v2/MORPH_Album2-izfnet-simplenn/new_5k.mat') ;
% load('model_v2/MORPH_Album2-izfnet-simplenn/new_5k_indices.mat') ;
% test_id = find(indices ==1);
% val_labels =  age(test_id');


% indices = crossvalind('kfold',age,10);



 
load(fullfile(root_path,'fc7_feat.mat'))

cnn_feats = feats';
cnn_feats = test_fea;
cnn_feats = fea;
cnn_feats = data(test_id,:);
% cnn_feats = predictions';

%l2 normalization
for i = 1:size(cnn_feats,1)
    cnn_feats(i,:) = cnn_feats(i,:)./norm(cnn_feats(i,:));
end
% Set parameters
no_dims = 2;
initial_dims = 50;
perplexity = 30;
% Run t−SNE
mappedX = tsne(cnn_feats, [], no_dims, initial_dims, perplexity);
% Plot results
% gscatter(mappedX(:,2)-1, mappedX(:,1), val_labels(1,:));
% save(fullfile(root_path,'t-mappedX.mat'),'mappedX');
% load(fullfile(root_path,'t-mappedX.mat'))
%  save(fullfile(root_path,'ldl_mappedX.mat'),'mappedX');
load(fullfile(root_path,'ldl_mappedX.mat'))
% load(fullfile(root_path,'new_mappedX.mat'))
%% load embedding

% load('imagenet_val_embed.mat'); % load x (the embedding 2d locations from tsne)
x = mappedX;
x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));
gscatter(x(:,2), 1-x(:,1), val_labels(1,:),[],'.',15,'off');
% gscatter(x,y,g,clr,sym,siz,doleg,xnam,ynam)
set(gca,'ytick',[]);
set(gca,'xtick',[]);
box off
axis off
print(gcf,'-depsc',['figure/','morph_age.eps'])
print(gcf,'-depsc',['figure/','aflw_yaw_hog.eps'])
print(gcf,'-depsc',['figure/','aflw_pitch_hog.eps'])
print(gcf,'-depsc',['figure/','chalearn_age.eps'])
print(gcf,'-depsc',['figure/','chalearn_age_bif.eps'])
print(gcf,'-depsc',['figure/','morph_age_bif.eps'])

print(gcf,'-depsc',['figure/','aflw_yaw.eps'])
print(gcf,'-depsc',['figure/','aflw_pitch.eps'])



%% load validation image filenames
id = find(imdb.images.set==2);
% image_paths = fullfile(imdb.images.crop_imageDir,imdb.images.crop_name(id));

image_paths = fullfile(imdb.imageDir,imdb.images.name(id));
fs = image_paths;
N = length(fs);
fs'
%% create an embedding image

S = 4000; % size of full embedding image
G = zeros(S, S, 3, 'uint8');
s = 64; % size of every single image

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
    if G(a,b,1) ~= 0
        continue % spot already filled
    end
    
    I = imread(fullfile(fs{i}));
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);
    
    G(a:a+s-1, b:b+s-1, :) = I;
    
end
% imwrite(G, 'alfw_dldl_64.jpg', 'jpg');
imwrite(G, 'chalearn_dldl_64.jpg', 'jpg');

%  imwrite(G, 'figure/chalearn_dldl_48.jpg', 'jpg');
%  imwrite(G, 'figure/morph_dldl_48.jpg', 'jpg');
% % imshow(G);
% imwrite(G, 'figure/alfw_dldl.jpg_48', 'jpg');
imwrite(G, 'chalearn_dldl.jpg', 'jpg');
% imwrite(G, 'morph_dldl.jpg', 'jpg');
imwrite(G, 'alfw_dldl.jpg_48', 'jpg');
%%
%% average up images
% % (doesnt look very good, failed experiment...)
% 
% S = 1000;
% G = zeros(S, S, 3);
% C = zeros(S, S, 3);
% s = 50;
% 
% Ntake = 5000;
% for i=1:Ntake
%     
%     if mod(i, 100)==0
%         fprintf('%d/%d...\n', i, Ntake);
%     end
%     
%     % location
%     a = ceil(x(i, 1) * (S-s-1)+1);
%     b = ceil(x(i, 2) * (S-s-1)+1);
%     a = a-mod(a-1,s)+1;
%     b = b-mod(b-1,s)+1;
%     
%     I = imread(fs{i});
%     if size(I,3)==1, I = cat(3,I,I,I); end
%     I = imresize(I, [s, s]);
%     
%     G(a:a+s-1, b:b+s-1, :) = G(a:a+s-1, b:b+s-1, :) + double(I);
%     C(a:a+s-1, b:b+s-1, :) = C(a:a+s-1, b:b+s-1, :) + 1;
%     
% end
% 
% G(C>0) = G(C>0) ./ C(C>0);
% G = uint8(G);
% imshow(G);

%% do a guaranteed quade grid layout by taking nearest neighbor

S = 2000; % size of final image
G = zeros(S, S, 3, 'uint8');
s = 50; % size of every image thumbnail

xnum = S/s;
ynum = S/s;
used = false(N, 1);

qq=length(1:s:S);
abes = zeros(qq*2,2);
i=1;
for a=1:s:S
    for b=1:s:S
        abes(i,:) = [a,b];
        i=i+1;
    end
end
%abes = abes(randperm(size(abes,1)),:); % randperm

for i=1:size(abes,1)
    a = abes(i,1);
    b = abes(i,2);
    %xf = ((a-1)/S - 0.5)/2 + 0.5; % zooming into middle a bit
    %yf = ((b-1)/S - 0.5)/2 + 0.5;
    xf = (a-1)/S;
    yf = (b-1)/S;
    dd = sum(bsxfun(@minus, x, [xf, yf]).^2,2);
    dd(used) = inf; % dont pick these
    [dv,di] = min(dd); % find nearest image

    used(di) = true; % mark as done
%     I = imread(fs{di});
    I = imread(fullfile(fs{i}));

    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);

    G(a:a+s-1, b:b+s-1, :) = I;

    if mod(i,100)==0
        fprintf('%d/%d\n', i, size(abes,1));
    end
end

imshow(G);

%%
imwrite(G, 'cnn_embed_full_2k.jpg', 'jpg');

