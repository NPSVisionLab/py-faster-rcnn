function rec = VOCreadxml(path)

if length(path)>5&&strcmp(path(1:5),'http:')
    xml=urlread(path)';
else
    display(path);
    f=fopen(path,'r');
    xml=fread(f,'*char')';
    fclose(f);
end
rec=VOCxml2struct(xml);
