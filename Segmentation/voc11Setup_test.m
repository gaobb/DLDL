
imdb.paths.image = '/home/gaobb/mywork/SV3/image_data/VOC2011_ALL/Test/VOCdevkit/VOC2011/JPEGImages/%s.jpg';
name_list = '/home/gaobb/mywork/SV3/image_data/VOC2011_ALL/Test/VOCdevkit/VOC2011/ImageSets/Segmentation/test.txt';
[names] = textread(name_list, '%s') ;
imdb.classes.id = uint8(1:20) ;
imdb.classes.name = {...
  'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', ...
  'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', ...
  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'} ;
imdb.classes.images = cell(1,20) ;
imdb.images.name = names;
