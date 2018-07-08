% --------------------------------------------------------
% Fast R-CNN
% Copyright (c) 2015 Microsoft
% Licensed under The MIT License [see LICENSE for details]
% Written by Ross Girshick
% --------------------------------------------------------

function showboxes(im, boxes,age)

%  image(im);
imshow(im,'border','tight','initialmagnification','fit');
[h,w,d] = size(im);
set (gcf,'Position',[0,0,w,h])

axis image;
axis off;
set(gcf, 'Color', 'white');

if ~isempty(boxes)
  x1 = boxes(:, 1);
  y1 = boxes(:, 3);
  x2 = boxes(:, 2);
  y2 = boxes(:, 4);
  c = 'g';
  s = '-';
  line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', ...
       'color', c, 'linewidth', 2, 'linestyle', s);
   text(double((x2(1)-x1(1))*0.5+x1(1)), double(y1(1)), ...
       sprintf('%d', age), ...
         'backgroundcolor', 'y', 'color', 'r','fontsize',25,'horiz','center','vertical','bottom');  
%   for i = 1:size(boxes, 1)
%     text(double(x1(i)), double(y1(i)) - 2, ...
%          sprintf('%.3f', boxes(i, end)), ...
%          'backgroundcolor', 'r', 'color', 'w');
%   end
end
