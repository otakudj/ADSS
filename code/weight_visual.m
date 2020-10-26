addpath('alfds');
for num=0:4
    X=dlmread(sprintf('alfdmodel%.2d.txt',num),' ',2,0);
    X=X(1,1:288);
    X=reshape(X,[18,16]);
    img=zeros(32,32);
    for i=1:16
        for j=1:18
            if(j<=16)
                row=floor((i-1)/4)*8+floor((j-1)/4)+3;
                col=floor(mod(i-1,4))*8+floor(mod(j-1,4))+3;
                img(row,col)=X(j,i);
            elseif(j==17)
                left_top_row=floor((i-1)/4)*8+2;
                left_top_col=floor(mod(i-1,4))*8+2;
                img(left_top_row:left_top_row+5,[left_top_col left_top_col+5])...
                    =X(j,i);
                img([left_top_row left_top_row+5],left_top_col:left_top_col+5)...
                    =X(j,i);
            else
                left_top_row=floor((i-1)/4)*8+1;
                left_top_col=floor(mod(i-1,4))*8+1;
                img(left_top_row:left_top_row+7,[left_top_col left_top_col+7])...
                    =X(j,i);
                img([left_top_row left_top_row+7],left_top_col:left_top_col+7)...
                    =X(j,i);
            end
        end
    end
    img_nom=(img+1)/2;
    figure();
    imshow(img_nom);
end
