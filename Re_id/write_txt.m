name={'TUD-Stadtmitte','TUD-Campus','PETS09-S2L1','ETH-Bahnhof',...
    'ETH-Sunnyday','ETH-Pedcross2','ADL-Rundle-6','ADL-Rundle-8',...
    'KITTI-13','KITTI-17','Venice-2'};

for i=1:length(name)
    contents=dir(['triplet-reid-master/det/',name{i}]);
    det=[];
    f=fopen(['triplet-reid-master/file/',name{i},'.txt'],'w');
    for j=1:length(contents)-3
        det{j}=['det/',name{i},'/',sprintf('det%.6d.jpg',j)];
        fprintf(f,'%s\n',det{j});
    end
    fclose(f);
end