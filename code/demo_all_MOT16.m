function demo_all_MOT16
name={'MOT16-01','MOT16-06',...
    'MOT16-07','MOT16-08','MOT16-12','MOT16-14'};
if isempty(dir('../../MOT/Data/motchallenge-devkit/motchallenge/res/MOT16_test'))
    mkdir('../../MOT/Data/motchallenge-devkit/motchallenge/res/MOT16_test');
end
tic();
for h=1:length(name)
    fprintf('%%%%%%%%%%  processing %s %%%%%%%%%%\n',name{h});
    res=MOT16(name{h},'MOT16',0,0);
    dlmwrite(fullfile('../../MOT/Data/motchallenge-devkit/motchallenge/res/MOT16_test',[name{h},'.txt']),res,',');
end
toc();