
[status, list] = system( 'ls /home/tomb/git/CVAC/data/ak47_people/train/neg/*.jpg' );
result = textscan( list, '%s', 'delimiter', '\n' );
fileList = result{1};


addpath('/home/tomb/py-faster-rcnn/VOCdevkit/VOCcode');


for i = 1 :  length(fileList)
    rec = PASemptyrecord();
    rec.filename = fileList(i);
    filename = char(rec.filename(1));
    [pathstr, name, ext] = fileparts(filename);
    rec.filename = [name ext];
    myobj = '__background__';
    %myobj = 'background';
    rec.folder = myobj;
    newfile = regexprep(filename, '.jpg', '.xml');
    newfile = regexprep(newfile, 'neg', 'annotations_neg');
    image = imread(filename);
    [w, h, d] = size(image);
    clear('image')
    rec.size.width = w;
    rec.size.height = h;
    rec.size.depth = d;
    rec.segmented = 0;
    rec.source.database = 'CVAC';

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
    
