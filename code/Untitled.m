
name={'TUD-Stadtmitte','TUD-Campus','PETS09-S2L1','ETH-Bahnhof',...
    'ETH-Sunnyday','ETH-Pedcross2','ADL-Rundle-6','ADL-Rundle-8',...
    'KITTI-13','KITTI-17','Venice-2'};
for i=1:length(name)
    fun(name{i},'train',0)
end
    


function fun(name,pattern,visual_det)
warning off;
% % param.overlap=0.2;
param.re_id=17;
param.articulated_num=5;
param.articulated_rate=0.9;
% % param.del=0.8;
param.tao=40;
param.iter=10;
% global param;
% addpath(fullfile('det',pattern,name));
gt=dlmread(fullfile('E:\multiple object tracking\Data\2DMOT2015\2DMOT2015',pattern,name,'gt\gt.txt'));
det_dir=dir(fullfile('det',pattern,name));
det_len=length(det_dir);
% cd (fullfile(pattern,'train','det',name));
det=cell(1,det_len-2);
len=zeros(1,det_len-2);
for i=1:det_len-2
    if det_dir(i+2).bytes~=0
        det{i}=gt(gt(:,1)==i,3:6);
        det{i}(:,3:4)=det{i}(:,3:4)+det{i}(:,1:2);
        len(i)=size(det{i},1);
    end
end
% cd ../../../;


%% Articulated
del_re=[];
fprintf('Articulation...\n')
for i=1:length(len)
    del=[];
    X=load(fullfile('Articulated\pose-tensorflow-master\res',pattern,name,[sprintf('%.6d',i),'_x.txt']));
    Y=load(fullfile('Articulated\pose-tensorflow-master\res',pattern,name,[sprintf('%.6d',i),'_y.txt']));
    if ~isempty(X)
        coordi=reshape([X,Y],[size(X),2]);
        for j=1:size(X,1)
            zzz=find(X(j,:)~=0);
            if length(zzz)<param.articulated_num
                del(end+1)=j;
            end
        end
        coordi(del,:,:)=[];
        
        del=[];
        rate=[];
        for j=1:size(det{i},1)
            for k=1:size(coordi,1)
                if k>size(coordi,1)
                    break;
                end
                trans=reshape(coordi(k,:,:),[size(coordi,2),2]);
                trans(find(trans(:,1)==0),:)=[];
                rate(j,k)=length(find(sum([trans(:,1)>det{i}(j,1),...
                    trans(:,2)>det{i}(j,2),...
                    trans(:,1)<det{i}(j,3),...
                    trans(:,2)<det{i}(j,4)]...
                    ,2)==4))/length(trans(:,1));
                
            end
        end
        if isempty(rate)
            continue;
        end
        rate=(rate>param.articulated_rate).*rate;
        [val,ind]=max(rate,[],1);
        ind=unique(ind(find(val~=0)));
        del=setdiff(1:size(det{i},1),ind);
        if ~isempty(del)
            %         del(end+1)=j;
            del_re=[del_re,del+det_ind(i,1,len)-1];
        end
        if ~isempty(det{i})
            det{i}=det{i}(ind,:);
        end
        %     if val>param.articulated_rate
        %
        %         rate(ind)=[];
        %         %             if max(rate)<0.2;
        %         %                 num=length(find(trans(:,1)~=0));
        %         %             trans(find(trans(:,1)==0),:)=[];
        %         %             square=(det{i}(j,3)-det{i}(j,1))*(det{i}(j,4)-det{i}(j,2))/(((max(trans(:,1))-min(trans(:,1))))*((max(trans(:,2))-min(trans(:,2)))));
        %         %             if square<2
        %         temp=1;
        %         %                 coordi(ind,:,:)=[];
        %         %                 break;
        %         %             end
        %     end
    end
end


len=zeros(1,length(det));
for i=1:length(det)
    if det_dir(i+2).bytes~=0
        len(i)=size(det{i},1);
    end
end

%% Dectections Visualization
if visual_det==1
    k=1;
    for i=1:length(len)
        full_path=fullfile('../../MOT/Data/2DMOT2015',pattern,name,'img1');
        img_name=sprintf('%06d.jpg',i);
        img_path=fullfile(full_path,img_name);
        im=imread(img_path);
        %     set(gca,'Position',[0,0,1,1],'visible','off');
        %     figure(1);
        
        im_handle=imshow(im);
        hold on;
        
        for j=1:len(i)
            fx = det{i}(j,1)+2;
            fy = det{i}(j,2)+12;
            %             text_handle = text(fx,fy,int2str(k),'Fontsize',14,'color','red');
            rectangle('Position',[det{i}(j,1:2) det{i}(j,3:4)-det{i}(j,1:2)],'Edgecolor','b','LineWidth',3);
            k=k+1;
        end
        %         saveas(im_handle,['det/',sprintf('%.3d',i),'.jpg']);
        
        drawnow;
        pause(0.3);
        hold off;
    end
end
out=[];
for i=1:length(det)
    out(end+1:end+size(det{i},1),:)=[ones(size(det{i},1),1)*i,det{i}];
end
dlmwrite(['out/',name,'(gt).txt'],out)

end

function y=det_ind(frame,num,len)
y=0;
for i=1:frame-1
    y=y+len(i);
end
y=y+num;
end

function [fr,ind]=fr_ind(ind,len)
fr=1;
while ind-len(fr)>0
    ind=ind-len(fr);
    fr=fr+1;
end
end