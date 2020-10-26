name={'TUD-Stadtmitte','TUD-Campus','PETS09-S2L1','ETH-Bahnhof',...
    'ETH-Sunnyday','ETH-Pedcross2','ADL-Rundle-6','ADL-Rundle-8',...
    'KITTI-13','KITTI-17','Venice-2'};
num=length(name);
for i=1:num
    contents=dir(['det/',name{i}]);
    m=length(contents)-2;
%     if isempty(dir(['file/',name{i},'.txt']))
%         mkdir(['file/',name{i},'.txt']);
%     end
        
    fid=fopen(['file/',name{i},'.txt'],'w');
    for j=1:m
        fprintf(fid,['det/',name{i},'/','det%.6d.jpg\n'],j);
    end
    fclose(fid);
end
        
    