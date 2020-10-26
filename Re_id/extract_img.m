% name={'TUD-Stadtmitte','TUD-Campus','PETS09-S2L1','ETH-Bahnhof',...
%     'ETH-Sunnyday','ETH-Pedcross2','ADL-Rundle-6','ADL-Rundle-8',...
%     'KITTI-13','KITTI-17','Venice-2'};
content = dir('')
base_path='/home/cgv841/otakudj/MOT/Data/2DMOT2015/train/';


for i=1:11
    if isempty(dir(['/home/cgv841/otakudj/MyWork/MOT/Re_id/triplet-reid-master/det/',name{i}]))
        mkdir(['/home/cgv841/otakudj/MyWork/MOT/Re_id/triplet-reid-master/det/',name{i}])
    end
    full_path=fullfile(base_path,name{i},'img1');
    contents=dir(full_path);
    count=1;
    for j=1:length(contents)-2
        img=imread([full_path,sprintf('/%.6d',j),'.jpg']);
        imshow(img);
        hold on;
        [y,x,~]=size(img);
        det=load(['/home/cgv841/otakudj/MyWork/MOT/train/',name{i},sprintf('/det/det%.6d.txt',j)]);
        for k=1:size(det,1)
            x1=max(round(det(k,1)),1);
            y1=max(round(det(k,2)),1);
            x2=min(round(det(k,3)),x);
            y2=min(round(det(k,4)),y);
            rectangle('Position',[x1,y1,x2-x1,y2-y1],'EdgeColor','b')
            out=img(y1:y2,x1:x2,:);
            imwrite(out,sprintf(['triplet-reid-master/det/',name{i},'/det%.6d.jpg'],count));
            count=count+1;
        end
    end
end