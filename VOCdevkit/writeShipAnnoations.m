directory = '/media/tomb/newvol/images/ships/HLD_distro/tanker/tanker_toosmall'
[~, sysout] = system(['ls ' fullfile(directory,'*{jpeg,jpg,png}')]);
%[~, sysout] = system(['ls ' fullfile(directory,'*{jpeg,jpg}')]);
fileList = regexp(sysout,'\n','split');



addpath('/home/tomb/py-faster-rcnn/VOCdevkit/VOCcode');


for i = 1 :  length(fileList)
    rec = PASemptyrecord();
    rec.filename = fileList(i);
    filename = char(rec.filename(1));
    [pathstr, name, ext] = fileparts(filename);
    rec.filename = [name ext];
    myobj = 'tanker';
    rec.folder = myobj;
    newfile = fullfile(directory, [name '.xml'])
   
    image = imread(filename);
    [w, h, d] = size(image);
    clear('image')
    rec.size.width = w;
    rec.size.height = h;
    rec.size.depth = d;
    rec.segmented = 0;
    rec.source.database = 'HLD_distro';

    j = 1;
    rec.object(j).name = myobj;
    rec.object(j).bndbox.xmin = 1; 
    rec.object(j).bndbox.ymin = 1;
    rec.object(j).bndbox.xmax = w;
    rec.object(j).bndbox.ymax = h;
    rec.object(j).truncated = 0;
    rec.object(j).difficult = 0;
    
    
    VOCwritexml(rec, newfile)
    
end
    
