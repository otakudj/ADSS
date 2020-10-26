function res=metric_v1(name,pattern,visual_det,visual_res)
warning off;
param.overlap=0.2;
param.re_id=17;
param.articulated_num=5;
param.articulated_rate=0.7;
% % param.del=0.8;
param.tao=40;
param.iter=5;
% global param;
% addpath(fullfile('det',pattern,name));
det_dir=dir(fullfile('det',pattern,name));
det_len=length(det_dir);
% cd (fullfile(pattern,'train','det',name));
det=cell(1,det_len-2);
len=zeros(1,det_len-2);
for i=1:det_len-2
    if det_dir(i+2).bytes~=0
        det{i}=dlmread(fullfile('det',pattern,name,sprintf('det%.6d.txt',i)));
        len(i)=size(det{i},1);
    end
end
% cd ../../../;

%% Re_id
fprintf('%20.20s\t','Re-id calculation...')
tic
time = 0;
directory=fullfile('Re_id/triplet-reid-master/res',pattern);
fid=fopen([fullfile(directory,name),'.txt']);
i=1;
while ~feof(fid)
    line=fgetl(fid);
    p=textscan(line,'%f %*c');
    p=cell2mat(p);
    f(i,:)=p;
    i=i+1;
end
fclose(fid);

num=size(f,1);
re=zeros(num);
for i=1:num
    fr=fr_ind(i,len);
    ind=det_ind(fr+1,1,len);
    fr_end=min(fr+param.tao,length(len));
    for j=ind:det_ind(fr_end,len(fr_end),len)
        re(i,j)=(sum((f(i,:)-f(j,:)).^2)).^0.5;
    end
end
fprintf('%7.3f s\n',toc - time);
% p=cell(det_len,det_len);
% lam=20;
% for i=1:det_len
%     for j=i+1:det_len
%         if ~isempty(alfd{i,j})
%             pij=alfd{i,j};
%             pji=alfd{j,i};
%             p{i,j}=(pij+pji)/(sum(pij)+sum(pji)+lam);
%         end
%     end
% end
%
%
% addpath('alfds');
% W=zeros(5,288);
% for num=1:5
%     temp=dlmread(sprintf('alfdmodel%.2d.txt',num-1),' ',2,0);
%     W(5-num+1,:)=temp(1,1:288);
% end
%
% r=zeros(det_len,det_len);
% for i=1:det_len
%     for j=1:det_len
%         if ~isempty(p{i,j})
%             delta=fr_ind(j,len)-fr_ind(i,len);
%             switch(delta)
%                 case 1
%                     r(i,j)=W(1,:)*p{i,j};
%                 case 2
%                     r(i,j)=W(2,:)*p{i,j};
%                 case 5
%                     r(i,j)=W(3,:)*p{i,j};
%                 case 10
%                     r(i,j)=W(4,:)*p{i,j};
%                 case 20
%                     r(i,j)=W(5,:)*p{i,j};
%             end
%         end
%     end
% end
% sim = reshape(re,1,size(re,1) * size(re, 2));
% [val_hist, ind_hist] = hist(sim, 10:0.1:20);
% [~, re_id] = min(val_hist);
% param.re_id = ind_hist(re_id);

%% Articulated
del_re=[];
fprintf('%20.20s\t','Articulation...')
time = toc;
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


f(del_re,:)=[];
re(del_re,:)=[];
re(:,del_re)=[];

len=zeros(1,length(det));
for i=1:length(det)
    if det_dir(i+2).bytes~=0
        len(i)=size(det{i},1);
    end
end
fprintf('%7.3f s\n',toc - time);

% %% NMS
% fprintf('%20.20s\t','NMS...')
% time = toc;
% pre_det=det;
% del_re=[];
% for i=1:length(pre_det)
%     del=[];
%     for j=1:size(pre_det{i},1)
%         p1=[pre_det{i}(j,1) pre_det{i}(j,2);
%             pre_det{i}(j,1) pre_det{i}(j,4);
%             pre_det{i}(j,3) pre_det{i}(j,2);
%             pre_det{i}(j,3) pre_det{i}(j,4)];
%         s1=(pre_det{i}(j,3)-pre_det{i}(j,1))*(pre_det{i}(j,4)-pre_det{i}(j,2));
%         for k=j+1:size(pre_det{i},1)
%             p2=[pre_det{i}(k,1) pre_det{i}(k,2);
%                 pre_det{i}(k,1) pre_det{i}(k,4);
%                 pre_det{i}(k,3) pre_det{i}(k,2);
%                 pre_det{i}(k,3) pre_det{i}(k,4)];
%             s2=(pre_det{i}(k,3)-pre_det{i}(k,1))*(pre_det{i}(k,4)-pre_det{i}(k,2));
%             x1=max(min(p1(:,1)),min(p2(:,1)));
%             x2=min(max(p1(:,1)),max(p2(:,1)));
%             y1=max(min(p1(:,2)),min(p2(:,2)));
%             y2=min(max(p1(:,2)),max(p2(:,2)));
%             if x2-x1<=0
%                 r=0;
%             else
%                 s12=(x2-x1)*(y2-y1);
%                 r1=s12/s1;
%                 r2=s12/s2;
%                 if r1>param.overlap || r2>param.overlap
%                     if pre_det{i}(j,6)<pre_det{i}(k,6)
%                         del(end+1)=j;
%                         del_re(end+1)=det_ind(i,j,len);
%                         break;
%                     else
%                         del(end+1)=j;
%                         del_re(end+1)=det_ind(i,j,len);
%                     end
%                 end
%             end
% 
%         end
%     end
%     if ~isempty(del)
%         pre_det{i}(del,:)=[];
%     end
% end
% re(del_re,:)=[];
% re(:,del_re)=[];
% 
% det=pre_det;
% len=zeros(1,length(det));
% for i=1:length(det)
%     if det_dir(i+2).bytes~=0
%         len(i)=size(det{i},1);
%     end
% end
% fprintf('%7.3f s\n',toc - time);
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


%% Minimax
fprintf('%20.20s\t','Minimax...');
time = toc;
det_len=sum(len);
l=zeros(1,det_len);
r(1:det_len)=inf;
s=1;
while ~len(s)
    s=s+1;
end
Q{1}=det_ind(s,1,len):det_ind(s,1,len)+len(s)-1;
r(Q{1})=0;
l(1:length(Q{1}))=Q{1};

m=1;
id=Q{1}(end)+1;
while(~isempty(Q{m}))
    Q{m+1}=[];
    for i=1:length(Q{m})
        N=[];
        fr=fr_ind(Q{m}(i),len);
        for t=fr+1:min(fr+param.tao,length(len))
            hd=det_ind(t,1,len);
            tl=hd+len(t)-1;
            [val,ind]=min(re(Q{m}(i),hd:tl));
            if ~isempty(val) && val<param.re_id
                N(end+1,1:2)=[hd+ind-1,val];
            end
        end
        for j=1:size(N,1)
            r_star=max(N(j,2),r(Q{m}(i)));
            r_dot=inf;
            
            fr=fr_ind(N(j,1),len);
            a=det_ind(fr,1,len);
            b=a+len(fr)-1;
            ind=find(l(a:b)==l(Q{m}(i)),1);
            if ~isempty(ind)
                ind_dot=a+ind-1;
                r_dot=r(ind_dot);
            end
            if r_star<r(N(j,1)) && r_star<r_dot
                r(N(j,1))=r_star;
                l(N(j,1))=l(Q{m}(i));
                if r_dot<inf
                    r(ind_dot)=inf;
                    l(ind_dot)=0;
                end
                Q{m+1}(end+1)=N(j,1);
            end
        end
    end
    k_dot=find(l==0,1);
    if isempty(Q{m+1}) && ~isempty(k_dot)
        l(k_dot)=id;
        r(k_dot)=0;
        Q{m+1}=k_dot;
        id=id+1;
    end
    m=m+1;
end
fprintf('%7.3f s\n',toc - time);

%         if(~isempty(N))
%             if isempty(find(label==ind,1)) && label(Q{m}(i))==-1
%                 label(Q{m}(i))=ind;
%                 Q{m+1}(end+1)=ind;
%             else
%                 pos=find(label==ind,1);
%                 if(re(pos,ind)>val)
%                     re(pos,ind)=inf;
%                     while label(pos)~=-1
%                         pre=pos;
%                         pos=label(pos);
%                         label(pre)=-1;
%                         if pos==0
%                             break;
%                         end
%                     end
%                     label(Q{m}(i))=ind;
%                     Q{m+1}(end+1)=ind;
%                 end
%             end
%         end
%     end
%     if(isempty(Q{m+1}) && ~isempty(find(label==-1, 1)))
%         k_pre=k;
%         k=find(label==-1,1);
%         if k~=k_pre
%             Q{m+1}(end+1)=k;
%         else
%             label(k)=0;
%             m=m-1;
%         end
%     end
%     m=m+1;
% end
label=l;
% l=1;
% flag=zeros(size(label));
% for i=1:det_len
%     k=i;
%     if flag(k)==0 && label(k)~=0
%         temp=1;
%         while label(k)~=0
%             flag(k)=1;
%             k_pre=k;
%             k=label(k_pre);
%             label(k_pre)=l;
%         end
%         flag(k)=1;
%         label(k)=l;
%         if temp==1
%             l=l+1;
%         end
%         temp=0;
%     end
%
% end



%% labels to traklets
for i=1:max(label)
    trajectory{i}.num=i;
    ind=find(label==i);
    trajectory{i}.ind=ind;
    trajectory{i}.det=[];
    for j=1:length(ind)
        [fr,num]=fr_ind(ind(j),len);
        trajectory{i}.det(end+1,1:8)=[fr i det{fr}(num,[1:4,6]) ind(j)];
    end
end



%% stitching tracklets
iter=0;
% stitch=[];
fprintf('%20.20s\n','Stitching...');
time = toc;
while iter<param.iter
    str = sprintf('%s%d(%d)...','iter',iter+1,length(trajectory));
    fprintf('%20.20s\t',str);
    time = toc;
    stitch1=[];
    for i=1:length(trajectory)
        temp=param.re_id;
        for j=i+1:length(trajectory)
            judge=1;
            for k=1:length(trajectory{i}.ind)
                for l=1:length(trajectory{j}.ind)
                    if fr_ind(trajectory{i}.ind(k),len)==fr_ind(trajectory{j}.ind(l),len)
                        judge=0;
                        break;
                    end
                end
                if ~judge
                    break;
                end
            end
            if judge
                %             %         if fr_ind(trajectory{i}.ind(end),len)<fr_ind(trajectory{j}.ind(1),len)
                for k=1:length(trajectory{i}.ind)
                    for l=1:length(trajectory{j}.ind)
                        if re(trajectory{i}.ind(k),trajectory{j}.ind(l)) && ...
                                re(trajectory{i}.ind(k),trajectory{j}.ind(l))<temp
                            temp=re(trajectory{i}.ind(k),trajectory{j}.ind(l));
                            link=[i,j,temp];
                        end
                    end
                end
                %             %         end
%                 ti=mean(f(trajectory{i}.ind,:));
%                 tj=mean(f(trajectory{j}.ind,:));
%                 dis=(sum(ti-tj).^2).^0.5;
%                 if dis<temp
%                     temp=dis;
%                     link=[i,j];
%                 end
            end
        end
        if temp~=param.re_id
            stitch1(end+1,1:3)=link;
            %         stitch1(end+1,1:3)=[link,temp];
        end
    end
    
    stitch2=[];
    for i=1:length(trajectory)
        temp=param.re_id;
        for j=i-1:-1:1
            judge=1;
            for k=1:length(trajectory{i}.ind)
                for l=1:length(trajectory{j}.ind)
                    if fr_ind(trajectory{i}.ind(k),len)==fr_ind(trajectory{j}.ind(l),len)
                        judge=0;
                        break;
                    end
                end
                if ~judge
                    break;
                end
            end
            if judge
                %                 if fr_ind(trajectory{j}.ind(end),len)<fr_ind(trajectory{i}.ind(1),len)
                %                     if isempty(intersect(trajectory{j}.ind,trajectory{i}.ind))
                for k=1:length(trajectory{i}.ind)
                    for l=1:length(trajectory{j}.ind)
                        if re(trajectory{j}.ind(l),trajectory{i}.ind(k)) && ...
                                re(trajectory{j}.ind(l),trajectory{i}.ind(k))<temp
                            temp=re(trajectory{j}.ind(l),trajectory{i}.ind(k));
                            link=[j,i,temp];
                        end
                    end
                end
                %                     end
%                 ti=mean(f(trajectory{i}.ind,:));
%                 tj=mean(f(trajectory{j}.ind,:));
%                 dis=(sum(ti-tj).^2).^0.5;
%                 if dis<temp
%                     temp=dis;
%                     link=[j,i];
%                 end
            end
        end
        if temp~=param.re_id
            stitch2(end+1,1:3)=link;
            %         stitch2(end+1,1:3)=[link,temp];
        end
    end
    stitch=intersect(stitch1,stitch2,'rows');
    if isempty(stitch)
        fprintf('%7.3f s\n',toc - time);
        break;
    end
    i=1;
    while i<=size(stitch,1)
        temp=1;
        j=i+1;
        while j<=size(stitch,1);
            if stitch(i,2)==stitch(j,1)
                if stitch(i,3)<=stitch(j,3)
                    stitch(j,:)=[];
                else
                    stitch(i,:)=[];
                    temp=0;
                    break;
                end
            end
            j=j+1;
        end
        if temp~=0
            i=i+1;
        end
    end
%     num=size(stitch,1);
%     stitch=mat2cell(stitch,ones(1,num));
%     i=1;
%     while i<=length(stitch)
%         temp=stitch{i}(2);
%         j=i+1;
%         while j<=length(stitch)
%             if stitch{j}(1)==temp
%                 stitch{i}(end+1)=stitch{j}(end);
%                 temp=stitch{i}(end);
%                 stitch(j)=[];
%             else
%                 j=j+1;
%             end
%         end
%         i=i+1;
%     end
    
%     for i=1:length(stitch)
%         for j=2:length(stitch{i})
%             if ~isempty(trajectory{stitch{i}(j)})
%                 trajectory{stitch{i}(1)}.ind=[trajectory{stitch{i}(1)}.ind,trajectory{stitch{i}(j)}.ind];
%                 trajectory{stitch{i}(1)}.det=[trajectory{stitch{i}(1)}.det;trajectory{stitch{i}(j)}.det];
%                 trajectory{stitch{i}(j)}=[];
%             end
%         end
%     end
    for i=1:size(stitch,1)
        trajectory{stitch(i,1)}.ind=[trajectory{stitch(i,1)}.ind,trajectory{stitch(i,2)}.ind];
        trajectory{stitch(i,1)}.det=[trajectory{stitch(i,1)}.det;trajectory{stitch(i,2)}.det];
        trajectory{stitch(i,2)}=[];
    end
    
    i=1;
    while i<=length(trajectory)
        if isempty(trajectory{i})
            trajectory(i)=[];
        else
            trajectory{i}.num=i;
            trajectory{i}.det(:,2)=ones(size(trajectory{i}.det,1),1)*i;
            i=i+1;
        end
    end
    
    fprintf('%7.3f s\n',toc - time);
    iter=iter+1;
end

    %     %%
    %     results=[];
    %     for i=1:length(trajectory)
    %         results=[results;trajectory{i}.det];
    %     end
    %     results=sortrows(results,1);
    %     results(:,5:6)=results(:,5:6)-results(:,3:4);
    %     results(:,9:10)=-1;
    %     if visual_res==1
    %         full_path=fullfile('../../MOT/Data/2DMOT2015/',pattern,name,'img1');
    %         img_contents=dir(full_path);
    %         img_numel=numel(img_contents)-2;
    %         j=1;
    %         for i=1:img_numel
    %             tic()
    %             img_name=sprintf('%06d.jpg',i);
    %             img_path=fullfile(full_path,img_name);
    %             im=imread(img_path);
    %             %     set(gca,'Position',[0,0,1,1],'visible','off');
    %             %     figure(1);
    %
    %             im_handle=imshow(im);
    %             hold on;
    %
    %             while(results(j,1)==i)
    %                 rectangle('Position',results(j,3:6),'Edgecolor','b');
    %                 fx = results(j,3)+ 2;
    %                 fy = results(j,4)+ 12;
    %                 text_handle = text(fx,fy,int2str(results(j,2)),'Fontsize',14,'color','red');
    %                 j=j+1;
    %                 if(j>size(results,1))
    %                     break;
    %                 end
    %             end
    %             pause(0.3);
    %             %         saveas(im_handle,['res/',sprintf('%.3d',i),'.jpg']);
    %             drawnow;
    %             hold off;
    %             if(j>size(results,1))
    %                 break;
    %             end
    %         end
    %     end
    
    
% %% stitching tracklets
% fprintf('%20.20s\t','Stitching...');
% time = toc;
% stitch1=[];
% for i=1:length(trajectory)
%     temp=param.re_id;
%     for j=i+1:length(trajectory)
% %         if fr_ind(trajectory{i}.ind(end),len)<fr_ind(trajectory{j}.ind(1),len)
%             for k=1:length(trajectory{i}.ind)
%                 for l=1:length(trajectory{j}.ind)
%                     if re(trajectory{i}.ind(k),trajectory{j}.ind(l)) && ...
%                             re(trajectory{i}.ind(k),trajectory{j}.ind(l))<temp
%                         temp=re(trajectory{i}.ind(k),trajectory{j}.ind(l));
%                         link=[i,j];
%                     end
%                 end
%             end
% %         end
%     end
%     if temp~=param.re_id
%         stitch1(end+1,1:2)=link;
% %         stitch1(end+1,1:3)=[link,temp];
%     end
% end
% 
% stitch2=[];
% for i=1:length(trajectory)
%     temp=param.re_id;
%     for j=i-1:-1:1
% %         if fr_ind(trajectory{j}.ind(end),len)<fr_ind(trajectory{i}.ind(1),len)
%             for k=1:length(trajectory{i}.ind)
%                 for l=1:length(trajectory{j}.ind)
%                     if re(trajectory{j}.ind(l),trajectory{i}.ind(k)) && ...
%                             re(trajectory{j}.ind(l),trajectory{i}.ind(k))<temp
%                         temp=re(trajectory{j}.ind(l),trajectory{i}.ind(k));
%                         link=[j,i];
%                     end
%                 end
%             end
% %         end
%     end
%     if temp~=param.re_id
%         stitch2(end+1,1:2)=link;
% %         stitch2(end+1,1:3)=[link,temp];
%     end
% end
% stitch=intersect(stitch1,stitch2,'rows');
% 
% num=size(stitch,1);
% stitch=mat2cell(stitch,ones(1,num));
% i=1;
% while i<=length(stitch)
%     temp=stitch{i}(2);
%     j=i+1;
%     while j<=length(stitch)
%         if stitch{j}(1)==temp
%             stitch{i}(end+1)=stitch{j}(end);
%             temp=stitch{i}(end);
%             stitch(j)=[];
%         else
%             j=j+1;
%         end
%     end
%     i=i+1;
% end
% 
% for i=1:length(stitch)
%     for j=2:length(stitch{i})
%         trajectory{stitch{i}(1)}.ind=[trajectory{stitch{i}(1)}.ind,trajectory{stitch{i}(j)}.ind];
%         trajectory{stitch{i}(1)}.det=[trajectory{stitch{i}(1)}.det;trajectory{stitch{i}(j)}.det];
%         trajectory{stitch{i}(j)}=[];
%     end
% end
% 
% i=1;
% while i<=length(trajectory)
%     if isempty(trajectory{i})
%         trajectory(i)=[];
%     else
%         trajectory{i}.num=i;
%         trajectory{i}.det(:,2)=ones(size(trajectory{i}.det,1),1)*i;
%         i=i+1;
%     end
% end
%% Decontamination
fprintf('%20.20s\t','Decontamination...');
time = toc;
for i=1:length(trajectory)
    trajectory{i}.det=sortrows(trajectory{i}.det,8);
    j=1;
    while j<size(trajectory{i}.det,1)
        if trajectory{i}.det(j,1)==trajectory{i}.det(j+1,1)
            if trajectory{i}.det(j,7)<trajectory{i}.det(j+1,7)
                trajectory{i}.det(j,:)=[];
            else
                trajectory{i}.det(j+1,:)=[];
            end
        else
            j=j+1;
        end
    end
end
fprintf('%7.3f s\n',toc - time);
%

%% interpolation
fprintf('%20.20s\t','Interpolation...');
time = toc;
for i=1:length(trajectory)
    trajectory{i}.det=sortrows(trajectory{i}.det,8);
    xx=[];
    if ~isempty(trajectory{i}.det)
        xx=trajectory{i}.det(1,1);
        for j=2:size(trajectory{i}.det,1)
            
%             if (7/3230*(trajectory{i}.det(j,1)-trajectory{i}.det(j-1,1))+183/3230)*...
%                     re(trajectory{i}.det(j-1,8),trajectory{i}.det(j,8))<1
            if (trajectory{i}.det(j,1)-trajectory{i}.det(j-1,1))<10 && ...
            re(trajectory{i}.det(j-1,8),trajectory{i}.det(j,8))<param.re_id
                xx=[xx,trajectory{i}.det(j-1,1):trajectory{i}.det(j,1)];
            else
                xx=[xx,trajectory{i}.det(j,1)];
            end
        end
        xx=unique(xx);
    end
%     xx=trajectory{i}.det(1,1):trajectory{i}.det(end,1);
    if length(xx)>1
        yy=[];
        yy(:,1)=interp1(trajectory{i}.det(:,1),trajectory{i}.det(:,3),xx);
        yy(:,2)=interp1(trajectory{i}.det(:,1),trajectory{i}.det(:,4),xx);
        yy(:,3)=interp1(trajectory{i}.det(:,1),trajectory{i}.det(:,5),xx);
        yy(:,4)=interp1(trajectory{i}.det(:,1),trajectory{i}.det(:,6),xx);

        ppp=zeros(length(xx),1);
        con=zeros(length(xx),1);
        for k=1:length(xx)
            pos=find(trajectory{i}.det(:,1)==xx(k),1);
            if(~isempty(pos))
                ppp(k)=trajectory{i}.det(pos,8);
                con(k)=trajectory{i}.det(pos,7);
            end
        end
        trajectory{i}.det=[xx',i*ones(length(xx),1),yy,con,ppp];
    end
end
fprintf('%7.3f s\n',toc - time);
%% remove overlap
% for i=1:length(trajectory)
%     if isempty(trajectory{i})
%         continue;
%     end
%     for j=i+1:length(trajectory)
%         if isempty(trajectory{j})
%             continue;
%         end
%         [c,a,b]=intersect(trajectory{i}.det(:,1),trajectory{j}.det(:,1));
%         if isempty(c)
%             break;
%         end
%         a_det=trajectory{i}.det(a,3:6);
%         b_det=trajectory{j}.det(b,3:6);
%         r=param.del;
%         temp=0;
%         for k=1:size(a_det,1)
%             p1=[a_det(k,1) a_det(k,2);
%                 a_det(k,1) a_det(k,4);
%                 a_det(k,3) a_det(k,2);
%                 a_det(k,3) a_det(k,4)];
%             s1=(a_det(k,3)-a_det(k,1))*(a_det(k,4)-a_det(k,2));
%             p2=[b_det(k,1) b_det(k,2);
%                 b_det(k,1) b_det(k,4);
%                 b_det(k,3) b_det(k,2);
%                 b_det(k,3) b_det(k,4)];
%             s2=(b_det(k,3)-b_det(k,1))*(b_det(k,4)-b_det(k,2));
%             x1=max(min(p1(:,1)),min(p2(:,1)));
%             x2=min(max(p1(:,1)),max(p2(:,1)));
%             y1=max(min(p1(:,2)),min(p2(:,2)));
%             y2=min(max(p1(:,2)),max(p2(:,2)));
%             if x2-x1<=0
%                 r=0;
%                 break;
%             else
%                 s12=(x2-x1)*(y2-y1);
%                 r1=s12/s1;
%                 r2=s12/s2;
%                 temp=temp+(r1+r2)/2;
%             end
%         end
%         if r~=0 && temp/size(a_det,1)>r
% %             trajectory{i}.ind=union(trajectory{i}.ind,trajectory{j}.ind);
%             trajectory{i}.det=union(trajectory{i}.det,trajectory{j}.det,'rows');
%             del=[];
%             for k=min(trajectory{i}.det(:,1)):max(trajectory{i}.det(:,1))
%                 deal=find(trajectory{i}.det(:,1)==k);
%                 if length(deal)==2
%                     deal1=trajectory{i}.det(deal(1),:);
%                     deal2=trajectory{i}.det(deal(1),:);
%                     if deal1(7)==0 && deal2(7)==0
%                         trajectory{i}.det(deal(1),:)=(deal1+deal2)/2;
%                         del(end+1)=deal(2);
%                     elseif deal1(7)==0
%                         del(end+1)=deal(1);
%                     else
%                         del(end+1)=deal(2);
%                     end
%                 end
%             end
%             trajectory{j}=[];
%             trajectory{i}.det(del,:)=[];
%             trajectory{i}.ind=trajectory{i}.det(trajectory{i}.det(:,7)~=0,7);
%         end
%     end
% end
% i=1;
% while i<=length(trajectory)
%     if isempty(trajectory{i})
%         trajectory(i)=[];
%     else
%         trajectory{i}.num=i;
%         trajectory{i}.det(:,2)=i;
%         i=i+1;
%     end
% end
%
%% results

results=[];
for i=1:length(trajectory)
    results=[results;trajectory{i}.det];
end
results=sortrows(results,1);
results(:,5:6)=results(:,5:6)-results(:,3:4);
results(:,9:10)=-1;

res=results;
res(:,7:8)=-1;




dlmwrite(['../../MOT/Data/motchallenge-devkit/motchallenge/res/temp/',name,'.txt'],res,',');

% base_path='../../2DMOT2015/train';
% contents=dir(base_path);
% names={};
% for i=3:numel(contents)
%     names{i-2}=contents(i).name;
% end
% choice=listdlg('ListString',names,'Name','Choose Videos','SelectionMode','single');

%% Results Visualizaiton
if visual_res==1
    full_path=fullfile('../../MOT/Data/2DMOT2015/',pattern,name,'img1');
    img_contents=dir(full_path);
    img_numel=numel(img_contents)-2;
    j=1;
    color = rand(50, 3);
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
            rectangle('Position',results(j,3:6),'Edgecolor',color(mod(results(j,2),50) + 1,:),'LineWidth',3);
%             fx = results(j,3)+ 2;
%             fy = results(j,4)+ 12;
%             text_handle = text(fx,fy,int2str(results(j,2)),'Fontsize',20,'color','red');
            j=j+1;
            if(j>size(results,1))
                break;
            end
        end
        pause(0.3);
%         saveas(im_handle,['res/',sprintf('%.3d',i),'.jpg']);
%         saveas(im_handle,['C:\Users\otakudj\Desktop\cvpr18\(17.10.22Yu)mypaper_v0.2\figs\exp_sti\',sprintf('%.4d',i),'.png']);
        drawnow;
        hold off;
        if(j>size(results,1))
            break;
        end
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