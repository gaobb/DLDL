function obj=matPlot(a, opt);
% matPlot: Display a matrix in a figure window
%
%	Usage:
%		matPlot(mat, opt);
%
%	Example:
%		% === Example 1
%		opt=matPlot('defaultOpt');
%		opt.matName='Magic matrix of size 10';
%		figure; matPlot(magic(8), opt);
%		% === Example 2
%		opt=matPlot('defaultOpt');
%		opt.matName='Random matrix of size 5';
%		opt.format='8.2f';
%		figure; matPlot(randn(4), opt);
%		% === Example 3
%		opt=matPlot('defaultOpt');
%		opt.showRowRightLabel=0;
%		opt.showColDownLabel=0;
%		opt.highlightDiagonal=0;
%		opt.matName='Months of the year';
%		opt.rowLabel={'Q1', 'Q2', 'Q3', 'Q4'};
%		opt.colLabel={'M1', 'M2', 'M3'};
%		mat={'Jan', 'Feb', 'Mar'; 'Apr', 'May', 'Jun'; 'Jul', 'Aug', 'Sept'; 'Oct', 'Nov', 'Dec'};
%		figure; matPlot(mat, opt);
%
%	Roger Jang, 20071009, 20120120

if nargin<1, selfdemo; return; end
% ====== Set the default options
if ischar(a) && strcmpi(a, 'defaultOpt')
	obj.matrixName='';
	obj.gridColor='k';
	obj.fontSize=10;
	obj.fontColor='a';
	obj.rowLabel=[];
	obj.colLabel=[];
	obj.highlightDiagonal=1;
	obj.showRowLeftLabel=1;
	obj.showRowRightLabel=1;
	obj.showColUpLabel=1;
	obj.showColDownLabel=1;
	obj.rowLeftLabel='';
	obj.rowRightLabel='';
	obj.colUpLabel='';
	obj.colDownLabel='';
	obj.format='11.4g';
	return
end
if nargin<2||isempty(opt), opt=feval(mfilename, 'defaultOpt'); end
if nargin<3, plotOpt=0; end

% Clear the current axis
cla reset;
[m,n]=size(a);

if opt.highlightDiagonal
	for i=1:min(m, n);
		patch(i-1+[0 1 1 0 0], min(m,n)-i+[0 0 1 1 0], 'y');
	end
end

% Place the text in the correct locations
for i=1:m		% Index over number of rows
	for j=1:n	% Index over number of columns
		theStr=a(i,j);
		if isnumeric(a(i,j))
			theStr=num2str(a(i,j), ['%', opt.format]);
		end
		obj.element(i,j)=text(j-.5, m-i+.5, theStr, ...
			'HorizontalAlignment', 'center', ...
			'Color', 'b', ...
			'FontWeight', 'bold', ...
			'FontSize', opt.fontSize);
	end
end

% ====== Fill row labels and row sum
for i=1:m
	if opt.showRowLeftLabel
		if isempty(opt.rowLeftLabel)
			for p=1:m, opt.rowLeftLabel{p}=int2str(p); end
		end
		obj.rowLeftLabel(i)=text(-0.1, m-i+.5, opt.rowLeftLabel{i}, ...
			'HorizontalAlignment', 'right', ...
			'Color', 'r', ...
			'FontWeight', 'bold', ...
			'FontSize', opt.fontSize);
	end
	if opt.showRowRightLabel
		if isempty(opt.rowRightLabel)
			if isnumeric(a)
				rowSum=sum(a, 2);
				for p=1:m, opt.rowRightLabel{p}=num2str(rowSum(p), ['%', opt.format]); end
			end
		end
		if ~isempty(opt.rowRightLabel)
			obj.rowRightLabel(i)=text(n+0.5, m-i+.5, opt.rowRightLabel{i}, ...
				'HorizontalAlignment', 'center', ...
				'Color', 'b', ...
				'FontWeight', 'bold', ...
				'FontSize', opt.fontSize);
		end
	end
end

% ====== Fill column labels and column sum
for j=1:n
	if opt.showColUpLabel
		if isempty(opt.colUpLabel)
			for p=1:n, opt.colUpLabel{p}=int2str(p); end
		end
		obj.colUpLabel(j)=text(j-.5, m+.1, opt.colUpLabel{j}, ...
			'HorizontalAlignment', 'left', ...
			'rot', 45, ...
			'Color', 'r', ...
			'FontWeight', 'bold', ...
			'FontSize', opt.fontSize);
	end
	if opt.showColDownLabel
		if isempty(opt.colDownLabel)
			if isnumeric(a)
				colSum=sum(a, 1);
				for p=1:m, opt.colDownLabel{p}=num2str(colSum(p), ['%', opt.format]); end
			end
		end
		if ~isempty(opt.colDownLabel)
			obj.colDownLabel(j)=text(j-.5, -.2, opt.colDownLabel{j}, ...
				'HorizontalAlignment', 'center', ...
				'Color', 'b', ...
				'FontWeight', 'bold', ...
				'FontSize', opt.fontSize);
		end
	end
end

set(gca,'Box', 'on', ...
        'Visible', 'on', ...
        'xLim', [0 n], ...
        'xGrid', 'on', ...
        'xTickLabel', [], ...
        'xTick', 0:n, ...
        'yGrid', 'on', ...
        'yLim', [0 m], ...
        'yTickLabel', [], ...
        'yTick', 0:m, ...
        'DataAspectRatio', [1, 1, 1], ... 
        'GridLineStyle', ':', ...
        'LineWidth', 3, ...
        'XColor', opt.gridColor, ...
        'YColor', opt.gridColor);

xlabel(opt.matrixName);
set(get(gca, 'xlabel'), 'position', [n/2, -1])
set(gcf, 'numbertitle', 'off', 'name', opt.matrixName);

% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);
