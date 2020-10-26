function res=metric(name,visual_det,visual_res)
param.overlap=0.2;
param.metric=0;
param.articulated=5;

addpath(fullfile('train',name,'det'));
det_dir=dir(fullfile('train',name,'alfd'));
det_len=length(det_dir);
cd (fullfile('train',name,'det'));
det=cell(1,det_len-2);
len=zeros(1,det_len-2);
for i=1:det_len-2
    if det_dir(i+2).bytes~=0
        det{i}=dlmread(sprintf('det%.6d.txt',i));
        len(i)=size(det{i},1);
    end
end
cd ../../../;

%% ALFD
directory=dir(fullfile('train',name,'alfd'));
fr_len=length(directory);
cd (fullfile('train',name,'alfd'));
det_len=sum(len);
alfd=cell(det_len,det_len);
for i=1:fr_len-2
    if directory(i+2).bytes~=0
        fid=fopen(sprintf('%.8d.txt',i-1));
        while ~feof(fid)
            line=fgetl(fid);
            p=textscan(line,'%d %*c');
            p=cell2mat(p);
            det1=det_ind(p(1)+1,p(2)+1,len);
            det2=det_ind(p(3)+1,p(4)+1,len);
            fea=zeros(p(5),1);
            for j=6:2:length(p)-1
                fea(p(j)+1)=p(j+1);
            end
            alfd{det1,det2}=fea;
        end
        fclose(fid);
    end
end
cd ../../../;

p=cell(det_len,det_len);
lam=20;
for i=1:det_len
    for j=i+1:det_len
        if ~isempty(alfd{i,j})
            pij=alfd{i,j};
            pji=alfd{j,i};
            p{i,j}=(pij+pji)/(sum(pij)+sum(pji)+lam);
        end
    end
end


addpath('alfds');
W=zeros(5,288);
for num=1:5
    temp=dlmread(sprintf('alfdmodel%.2d.txt',num-1),' ',2,0);
    W(5-num+1,:)=temp(1,1:288);
end

a=zeros(det_len,det_len);
for i=1:det_len
    for j=1:det_len
        if ~isempty(p{i,j})
            delta=fr_ind(j,len)-fr_ind(i,len);
            switch(delta)
                case 1
                    a(i,j)=W(1,:)*p{i,j};
                case 2
                    a(i,j)=W(2,:)*p{i,j};
                case 5
                    a(i,j)=W(3,:)*p{i,j};
                case 10
                    a(i,j)=W(4,:)*p{i,j};
                case 20
                    a(i,j)=W(5,:)*p{i,j};
            end
        end
    end
end



%% Articulated
del_a=[];
for i=1:length(len)
    del=[];
    X=load(fullfile('train',name,'Articulated',[sprintf('%.6d',i),'_x.txt']));
    Y=load(fullfile('train',name,'Articulated',[sprintf('%.6d',i),'_y.txt']));
    coordi=reshape([X,Y],[size(X),2]);
    for j=1:size(X,1)
        zzz=find(X(j,:)~=0);
        if length(zzz)<param.articulated
            del(end+1)=j;
        end
    end
    coordi(del,:,:)=[];
    
    del=[];
    for j=1:size(det{i},1)
        temp=0;
        for k=1:size(coordi,1)
            if k>size(coordi,1)
                break;
            end

            trans=reshape(coordi(k,:,:),[size(coordi,2),2]);
            rate=length(find(sum([trans>det{i}(j,1:2) trans<det{i}(j,3:4)],2)==4))/length(find(trans(:,1)~=0));
            if rate>0.8
                temp=1;
                coordi(k,:,:)=[];
                break;
            end
        end
        if temp==0
            del(end+1)=j;
            del_a(end+1)=det_ind(i,j,len);
        end
    end
    if ~isempty(det{i})
        det{i}(del,:)=[];
    end
end

a(del_a,:)=[];
a(:,del_a)=[];

len=zeros(1,length(det));
for i=1:length(det)
    if det_dir(i+2).bytes~=0
        len(i)=size(det{i},1);
    end
end


%% NMS
pre_det=det;
del_a=[];
for i=1:length(pre_det)
    del=[];
    for j=1:size(pre_det{i},1)
        p1=[pre_det{i}(j,1) pre_det{i}(j,2);
            pre_det{i}(j,1) pre_det{i}(j,4);
            pre_det{i}(j,3) pre_det{i}(j,2);
            pre_det{i}(j,3) pre_det{i}(j,4)];
        s1=(pre_det{i}(j,3)-pre_det{i}(j,1))*(pre_det{i}(j,4)-pre_det{i}(j,2));
        for k=j+1:size(pre_det{i},1)
            p2=[pre_det{i}(k,1) pre_det{i}(k,2);
                pre_det{i}(k,1) pre_det{i}(k,4);
                pre_det{i}(k,3) pre_det{i}(k,2);
                pre_det{i}(k,3) pre_det{i}(k,4)];
            s2=(pre_det{i}(k,3)-pre_det{i}(k,1))*(pre_det{i}(k,4)-pre_det{i}(k,2));
            x1=max(min(p1(:,1)),min(p2(:,1)));
            x2=min(max(p1(:,1)),max(p2(:,1)));
            y1=max(min(p1(:,2)),min(p2(:,2)));
            y2=min(max(p1(:,2)),max(p2(:,2)));
            if x2-x1<=0
                r=0;
            else
                s12=(x2-x1)*(y2-y1);
                r1=s12/s1;
                r2=s12/s2;
                if r1>param.overlap || r2>param.overlap
                    if pre_det{i}(j,6)<pre_det{i}(k,6)
                        del(end+1)=j;
                        del_a(end+1)=det_ind(i,j,len);
                        break;
                    else
                        del(end+1)=j;
                        del_a(end+1)=det_ind(i,j,len);
                    end
                end
            end

        end
    end
    if ~isempty(del)
        pre_det{i}(del,:)=[];
    end
end
a(del_a,:)=[];
a(:,del_a)=[];

det=pre_det;
len=zeros(1,length(det));
for i=1:length(det)
    if det_dir(i+2).bytes~=0
        len(i)=size(det{i},1);
    end
end

% %% test
% testify=zeros(length(len));
% for i=1:length(len)
%     x_start=det_ind(i,1,len);
%     x=x_start:x_start+len(i)-1;
%     for j=1:length(len)
%         y_start=det_ind(j,1,len);
%         y=y_start:y_start+len(j)-1;
%         if isempty(find(a(x,y),1)~=0)
%             testify(i,j)=0;
%         else
%             testify(i,j)=1;
%         end
%     end
%
% end



%% Dectections Visualization
if visual_det==1
    for i=1:length(len)
        full_path=fullfile('/home/cgv841/otakudj/MOT/Data/2DMOT2015/train',name,'img1');
        img_name=sprintf('%06d.jpg',i);
        img_path=fullfile(full_path,img_name);
        im=imread(img_path);
        %     set(gca,'Position',[0,0,1,1],'visible','off');
        %     figure(1);
        
        im_handle=imshow(im);
        hold on;
        
        for j=1:len(i)
            rectangle('Position',[det{i}(j,1:2) det{i}(j,3:4)-det{i}(j,1:2)],'Edgecolor','b');
        end
        
        drawnow;
        pause(1);
        hold off;
    end
end


%% Minimax
det_len=sum(len);
label=zeros(1,det_len)-1;
r(1:det_len)=inf;
r(len(1)+1:end)=0;
s=1;
while ~len(s)
    s=s+1;
end
Q{1}=det_ind(s,1,len):det_ind(s,1,len)+len(s)-1;
m=1;
k=0;
while(~isempty(Q{m}))
    Q{m+1}=[];
    for i=1:length(Q{m})
        fr=fr_ind(Q{m}(i),len);
        N=[];
        for t=[1 2 5 10 20]
%         for t=1
            if fr+t<=length(len)
                hd=det_ind(fr+t,1,len);
                tl=det_ind(fr+t,1,len)+len(fr+t)-1;
                [val,ind]=max(a(Q{m}(i),hd:tl));
                ind=hd+ind-1;
                if val>param.metric
                    N(1:2,end+1)=[ind;val];
                end
            end
        end
        if(~isempty(N))
            [val,ind]=max(N(2,:));
            ind=N(1,ind);
            if isempty(find(label==ind,1)) && label(Q{m}(i))==-1
                label(Q{m}(i))=ind;
                Q{m+1}(end+1)=ind;
            else
                pos=find(label==ind,1);
                if(a(pos,ind)<val)
%                     a(pos,ind)=0;
                    while label(pos)~=-1
                        pre=pos;
                        pos=label(pos);
                        label(pre)=-1;
                        if pos==0
                            break;
                        end
                    end
                    label(Q{m}(i))=ind;
                    Q{m+1}(end+1)=ind;
                end
            end
        end
    end
    if(isempty(Q{m+1}) && ~isempty(find(label==-1, 1)))
        k_pre=k;
        k=find(label==-1,1);
        if k~=k_pre
            Q{m+1}(end+1)=k;
        else
            label(k)=0;
            m=m-1;
        end
    end
    m=m+1;
end

l=1;
flag=zeros(size(label));
for i=1:det_len
    k=i;
    if flag(k)==0 && label(k)~=0
        temp=1;
        while label(k)~=0
            flag(k)=1;
            k_pre=k;
            k=label(k_pre);
            label(k_pre)=l;
        end
        flag(k)=1;
        label(k)=l;
        if temp==1
            l=l+1;
        end
        temp=0;
    end
    
end

%% labels to traklets
for i=1:max(label)
    trajectory{i}.num=i;
    ind=find(label==i);
    trajectory{i}.ind=ind;
    trajectory{i}.det=[];
    for j=1:length(ind)
        [fr,num]=fr_ind(ind(j),len);
        trajectory{i}.det(end+1,1:7)=[fr i det{fr}(num,1:4) ind(j)];
    end
end

%% stitching tracklets
stitch1=[];
for i=1:length(trajectory)
    temp=0;
    for j=i+1:length(trajectory)
        if fr_ind(trajectory{i}.ind(end),len)<fr_ind(trajectory{j}.ind(1),len)
            for k=1:length(trajectory{i}.ind)
                for l=1:length(trajectory{j}.ind)
                    if a(trajectory{i}.ind(k),trajectory{j}.ind(l))>temp
                        temp=a(trajectory{i}.ind(k),trajectory{j}.ind(l));
                        link=[i,j];
                    end
                end
            end
        end
    end
    if temp~=0
        stitch1(end+1,1:2)=link;
    end
end

stitch2=[];
for i=1:length(trajectory)
    temp=0;
    for j=i-1:-1:1
        if fr_ind(trajectory{j}.ind(end),len)<fr_ind(trajectory{i}.ind(1),len)
            for k=1:length(trajectory{i}.ind)
                for l=1:length(trajectory{j}.ind)
                    if a(trajectory{j}.ind(l),trajectory{i}.ind(k))>temp
                        temp=a(trajectory{j}.ind(l),trajectory{i}.ind(k));
                        link=[j,i];
                    end
                end
            end
        end
    end
    if temp~=0
        stitch2(end+1,1:2)=link;
    end
end
stitch=intersect(stitch1,stitch2,'rows');

num=size(stitch,1);
stitch=mat2cell(stitch,ones(1,num));
i=1;
while i<=length(stitch)
    temp=stitch{i}(2);
    j=i+1;
    while j<=length(stitch)
        if stitch{j}(1)==temp
            stitch{i}(end+1)=stitch{j}(end);
            temp=stitch{i}(end);
            stitch(j)=[];
        else
            j=j+1;
        end
    end
    i=i+1;
end

for i=1:length(stitch)
    for j=2:length(stitch{i})
        trajectory{stitch{i}(1)}.ind=[trajectory{stitch{i}(1)}.ind,trajectory{stitch{i}(j)}.ind];
        trajectory{stitch{i}(1)}.det=[trajectory{stitch{i}(1)}.det;trajectory{stitch{i}(j)}.det];
        trajectory{stitch{i}(j)}=[];
    end
end

i=1;
while i<=length(trajectory)
    if isempty(trajectory{i})
        trajectory(i)=[];
    else
        trajectory{i}.num=i;
        i=i+1;
    end
end

%% interpolation
for i=1:length(trajectory)
    xx=fr_ind(trajectory{i}.ind(1),len):fr_ind(trajectory{i}.ind(end),len);
    yy=[];
    yy(:,1)=interp1(trajectory{i}.det(:,1),trajectory{i}.det(:,3),xx);
    yy(:,2)=interp1(trajectory{i}.det(:,1),trajectory{i}.det(:,4),xx);
    yy(:,3)=interp1(trajectory{i}.det(:,1),trajectory{i}.det(:,5),xx);
    yy(:,4)=interp1(trajectory{i}.det(:,1),trajectory{i}.det(:,6),xx);
    ppp=zeros(length(xx),1);
    for k=1:length(xx)
        pos=find(trajectory{i}.det(:,1)==xx(k),1);
        if(~isempty(pos))
            ppp(k)=trajectory{i}.det(pos,7);
        end
    end
    trajectory{i}.det=[xx',i*ones(length(xx),1),yy,ppp];
end

%% results

results=[];
for i=1:length(trajectory)
    results=[results;trajectory{i}.det];
end
results=sortrows(results,1);
results(:,5:6)=results(:,5:6)-results(:,3:4);
results(:,8:10)=-1;

res=results;
res(:,7)=-1;
% dlmwrite(fullfile('New Folder',['TUD-Campus','.txt']),res,',');
% base_path='../../2DMOT2015/train';
% contents=dir(base_path);
% names={};
% for i=3:numel(contents)
%     names{i-2}=contents(i).name;
% end
% choice=listdlg('ListString',names,'Name','Choose Videos','SelectionMode','single');

%% Results Visualizaiton
if visual_res==1
    full_path=fullfile('/home/cgv841/otakudj/MOT/Data/2DMOT2015/train',name,'img1');
    img_contents=dir(full_path);
    img_numel=numel(img_contents)-2;
    j=1;
    for i=1:img_numel
        tic()
        img_name=sprintf('%06d.jpg',i);
        img_path=fullfile(full_path,img_name);
        im=imread(img_path);
        %     set(gca,'Position',[0,0,1,1],'visible','off');
        %     figure(1);
        
        im_handle=imshow(im);
        hold on;
        
        while(results(j,1)==i)
            rectangle('Position',results(j,3:6),'Edgecolor','b');
            fx = results(j,3)+ results(j,5)-52;
            fy = results(j,4)+20;
            text_handle = text(fx,fy,int2str(results(j,2)),'Fontsize',14,'color','red');
            j=j+1;
            if(j>size(results,1))
                break;
            end
        end
%         pause(1);
        saveas(im_handle,['img/',sprintf('%.3d',i),'.jpg']);
        drawnow;
        hold off;
    end
end
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