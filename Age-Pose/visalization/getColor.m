function color = getColor(value, min, max)
 %cmap = get(groot,'defaultfigurecolormap');
 cmap = colormap('hsv');
 [m, ~] = size(cmap);
 row = round((value/(max-min))*(m-1)) + 1;
 color = cmap(row, :);  
end 
