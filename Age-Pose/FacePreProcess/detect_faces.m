startup;
clear 
clc

% This script will detect faces in a set of images
% The minimum face detection size is 36 pixels,
% the maximum size is the full image.

% ori_path = './data/sample_test_images/pascal_faces';
image_dir = '../../data/face_image/';
Dataset = {'google-faces','ChaLearn/Train','ChaLearn/Validation','FG-NET/images','MORPH_Album2','Adience'};
   
for i =1: 66
Dataset{1,i} = fullfile('Adience',num2str(i));
end
    
for N =1: numel(Dataset)
    dataset = Dataset{N};
    
    ori_path = strcat(image_dir,dataset);
    exp_path = fullfile('det_face_result',dataset);%./face_detection_results
    if ~exist(fullfile(exp_path) )
        mkdir(fullfile(exp_path) )
    end
    crop_path = fullfile(exp_path,'crop');
    det_path = fullfile(exp_path,'det');
    
    if ~exist(crop_path)
      mkdir(crop_path)
    end
     if ~exist(det_path)
      mkdir(det_path)
    end
    bbox_file = fullfile(exp_path,'bbox.txt');
    noface_file = fullfile(exp_path,'noface_list.txt');
    
    model_path = 'data/trained_models/face_detection/dpm_baseline.mat';

    image_list_file = fullfile(exp_path,'list.txt');
    
    if ~exist(image_list_file)
        image_names = struct2cell(dir(strcat(ori_path,'/*.jpg')));
        image_names = [image_names struct2cell(dir(strcat(ori_path,'/*.JPG')))];
        fipin = fopen(image_list_file,'a');
        fprintf(fipin,'%d\n',numel(image_names(1,:)));
        for i=1:numel(image_names(1,:))
            fprintf(fipin,'%s\n',[image_names{1,i}]);
        end
        fclose(fipin);
    end
    
    face_model = load(model_path);
    
    % lower detection threshold generates more detections
     detection_threshold = 0;
    
    % 0.3 or 0.2 are adequate for face detection.
    nms_threshold = 0.3;
    
    image_names = dir(fullfile(ori_path, '*.jpg'));
    fidin  = fopen(image_list_file,'rt');
    fidbbox = fopen( bbox_file,'w');
    fidnoface = fopen( noface_file,'w');
    
    num = fgetl(fidin);
    while ~feof(fidin)
        image_name = fgetl(fidin);
        image_path = fullfile(image_dir,dataset,image_name);
        %  image_path = '../../experiment/Age_Estimation/data/face_image/ChaLearn/Train/image_159.jpg';
        image = imread(image_path);
        [w,h,c] = size(image);
        
        ds = [];
        im =  image;
        [ds, bs] = process_face(im, face_model.model,  ...
            detection_threshold, nms_threshold);
                        
        result_path = fullfile(exp_path, image_name);
        
        if  isempty(ds)
            fprintf(fidnoface,'%s \n',image_name);
        else
            tds = min(max(round(ds(1,1:4)),1),[h,w,h,w]);
            %showboxes(image, tds);
            fprintf(fidbbox,'%s %d %d %d %d\n',image_name,(tds(1,[1,3,2,4])));% left right top down
            imwrite(imresize(image(tds(1,2):tds(1,4),tds(1,1):tds(1,3),:),[256 256]),fullfile(crop_path,image_name));
            
            showsboxes_face(im, ds(1,:), fullfile(det_path,image_name));
            disp(['Created ', fullfile(det_path,image_name)]);
        end
      
    end
    fclose(fidin);
    fclose(fidbbox);
    fclose(fidnoface);
    
    disp('All images processed');
end

% 
cmd_str =[ './selectbbox/selectbbox  '   noface_file '  '      dataset  '   '  fullfile(exp_path,'noface_bbox.txt')];
system(cmd_str);


fid =  fopen( fullfile(exp_path,'noface_bbox.txt'));
while ~feof(fid)
    tline=fgetl(fid);                
    
    pos = find(tline==' ');
    pos([0 diff(pos)]==1) =[];
    img_name = tline(1:pos(1)-1);
    box = str2num(tline(pos(1):end));
    
    img_path = fullfile(image_dir,dataset,img_name);
    img = imread(img_path);
    [w,h,c] = size(image);
    imshow(img)
    imwrite(img(box(1,3):box(1,4),box(1,1):box(1,2),:),fullfile(crop_path,img_name));
    showsboxes_face(img, box(1,[1,3,2,4]), fullfile(det_path,img_name));
    disp(['Created ', fullfile(det_path,img_name)]);
    pause;
end
fclose(fid);


