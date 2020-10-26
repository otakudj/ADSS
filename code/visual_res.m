clc;
clear;

base_path='../../2DMOT2015/train';
contents=dir(base_path);
names={};
for i=3:numel(contents)
    names{i-2}=contents(i).name;
end
choice1=listdlg('ListString',names,'Name','Choose Videos','SelectionMode','single');
full_path=fullfile(base_path,names{choice1},'img1');
img_contents=dir(full_path);
img_numel=numel(img_contents)-2;

contents=dir('res')
algor={};
for i=3:numel(contents)
    algor{i-2}=contents(i).name;
end
choice2=listdlg('ListString',algor,'Name','Choose Videos','SelectionMode','single');

%% Read result
res_path=fullfile('res',algor{choice2},[names{choice1},'.txt']);
res=load(res_path);
j=1;
time=0;
for i=1:img_numel
    tic()
    img_name=sprintf('%06d.jpg',i);
    img_path=fullfile(full_path,img_name);
    im=imread(img_path);
%     set(gca,'Position',[0,0,1,1],'visible','off');
%     figure(1);
    
    im_handle=imshow(im);
    hold on;

    while(res(j,1)==i)
        rectangle('Position',res(j,3:6),'Edgecolor','r');
        fx = res(j,3)+ res(j,5)-52;
        fy = res(j,4)+20;                                              
        text_handle = text(fx,fy,int2str(res(j,2)),'Fontsize',14,'color','red');
        j=j+1;
        if(j>size(res,1))
            break;
        end
    end

    drawnow;
    hold off;
end