function demo_all_test
name={'ADL-Rundle-1','ADL-Rundle-3','AVG-TownCentre',...
    'ETH-Crossing','ETH-Jelmoli','ETH-Linthescher','KITTI-16',...
    'KITTI-19','PETS09-S2L2','TUD-Crossing','Venice-1'};
if isempty(dir('../../MOT/Data/motchallenge-devkit/motchallenge/res/test'))
    mkdir('../../MOT/Data/motchallenge-devkit/motchallenge/res/test');
end
time = 0;
for h=1:length(name)
    fprintf('****** Processing %s *******\n',name{h});
    fprintf('--------------------------------------\n')
    res=metric_v1(name{h},'test',0,0);
    fprintf('\n\n\n')
    time = time + toc();
    dlmwrite(fullfile('../../MOT/Data/motchallenge-devkit/motchallenge/res/test',[name{h},'.txt']),res,',');
end
fprintf('Total:%.3f', time);