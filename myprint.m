function myprint(filename, h)
% 'myprint' will produce vector eps and pdf files from a figure with no geometric change.
%  What you see will be what you get.
%
% Input:
%   filename: filename without suffix. e.g., 'figure1'
%   h: (optional) figure handle. If not specified, current figure handle will be used
%
% Example:
%   h = figure(1);
%   plot(1:50, (1:50).^2, '-b'); 
%   myprint('test', h); % generate test.eps and test.pdf.
%
% -----------------
% Wotao Yin (2011).

% if ~ischar(filename)
%     error('Input 1 must be char');
% end
% 
% if ~exist('h','var') || isempty(h)
%     h = gcf;
% end

ppos = get(h,'Position'); ppos(1:2)=0; psize = ppos(3:4);
dpi = get(0,'ScreenPixelsPerInch');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', psize/dpi);
set(gcf, 'PaperPosition', ppos/dpi);

%print('-painters','-depsc2','-r0',filename);
print('-painters','-dpdf','-r0',filename);

end

