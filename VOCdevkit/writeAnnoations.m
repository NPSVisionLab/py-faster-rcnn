ak = load('/home/tomb/git/CVAC/data/ak47_people/ak47.mat');

addpath('/home/tomb/py-faster-rcnn/VOCdevkit/VOCcode');


for i = 1 : length(ak.ak47Instances)
    rec = PASemptyrecord();
    rec.filename = ak.ak47Instances(i).imageFilename;
    filename = rec.filename;
    [pathstr, name, ext] = fileparts(filename);
    rec.filename = [name ext];
    myobj = 'ak47';
    rec.folder = myobj;
    newfile = regexprep(filename, '.jpg', '.xml');
    newfile = regexprep(newfile, 'pos', 'annotations_pos');
    image = imread(filename);
    [w, h, d] = size(image);
    clear('image')
    rec.size.width = w;
    rec.size.height = h;
    rec.size.depth = d;
    rec.segmented = 0;
    rec.source.database = 'CVAC'

    for j = 1 : length(ak.ak47Instances(i).objectBoundingBoxes(:,1))
        rec.object(j).name = myobj;
        rec.object(j).bndbox.xmin = ak.ak47Instances(i).objectBoundingBoxes(j,1); 
        rec.object(j).bndbox.ymin = ak.ak47Instances(i).objectBoundingBoxes(j,2);
        rec.object(j).bndbox.xmax = rec.object(j).bndbox.xmin + ak.ak47Instances(i).objectBoundingBoxes(j,3) -1;
        rec.object(j).bndbox.ymax = rec.object(j).bndbox.ymin + ak.ak47Instances(i).objectBoundingBoxes(j,4) -1;
        rec.object(j).truncated = 0
        rec.object(j).difficult = 0
    end
    
    VOCwritexml(rec, newfile)
    
end
    