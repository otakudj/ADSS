function demo_all_train
% global param;
% param.overlap=0.2;
% param.re_id=20;
% param.articulated_num=5;
% param.articulated_rate=0.7;
% param.del=0.8;
% param.tao=50;

name={'TUD-Stadtmitte','TUD-Campus','PETS09-S2L1','ETH-Bahnhof',...
    'ETH-Sunnyday','ETH-Pedcross2','ADL-Rundle-6','ADL-Rundle-8',...
    'KITTI-13','KITTI-17','Venice-2'};
% for k=15:30
%     param.re_id=k;
% for i=7:9
%     param.articulated_num=i;
%     for j=2:8
%         param.articulated_rate=j/10;
        time = 0;
        if isempty(dir(['../../MOT/Data/motchallenge-devkit/motchallenge/res/','+train']))
%                 ,int2str(i),'_0.',int2str(j)]))
            mkdir(['../../MOT/Data/motchallenge-devkit/motchallenge/res/','+train'])
%                 ,int2str(i),'_0.',int2str(j)]);
        end
        for h=1:length(name)
            fprintf('****** Processing %s *******\n',name{h});
            fprintf('------------------------------------\n');
            res=metric_v1(name{h},'train',0,0);
%             dlmwrite(fullfile('../../MOT/Data/motchallenge-devkit/motchallenge/res','+train',[name{h},'.txt']),res,',');
%                 ['train',int2str(i),'_0.',int2str(j)],...
            fprintf('\n\n\n')
            time = time + toc();
        end
        
        fprintf('Total:%.3f', time);
        %     end
% end
% end