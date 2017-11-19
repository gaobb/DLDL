function ov = IOUs(bbgt, bb)
% compat: change coordinate order from [y1 x1 h w] to [x1 y1 x2 y2]
% bbgt(:,3:4) = bbgt(:,1:2) + bbgt(:,3:4);
% bb(:,3:4) = bb(:,1:2) + bb(:,3:4);

bi = [max(bb(:,1),bbgt(1)) , max(bb(:,2),bbgt(2)) , min(bb(:,3),bbgt(3)) , min(bb(:,4),bbgt(4))];
iw = bi(:,3)-bi(:,1)+1;
ih = bi(:,4)-bi(:,2)+1;
ia = iw.*ih;
ia(iw<0 | ih<0) = 0;
ua = (bb(:,3)-bb(:,1)+1).*(bb(:,4)-bb(:,2)+1)+...
    (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-ia;
ov = ia./ua;
