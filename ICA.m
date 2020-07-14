clear
clc
close all
%% 图片规格标准化
length=256; 
width=256;
pic1=imread('1.png'); 
pic1_c=imresize(pic1,[length width]);
pic1_gray=rgb2gray(pic1_c);
pic2=imread('2.png'); 
pic2_c=imresize(pic2,[length width]);
pic2_gray=rgb2gray(pic2_c);
pic3=imread('3.png'); 
pic3_c=imresize(pic3,[length width]);
pic3_gray=rgb2gray(pic3_c);
% 原始图象及直方图
figure
subplot(321),imshow(pic1_gray),title('原始图片a')
subplot(322),imhist(pic1_gray)
subplot(323),imshow(pic2_gray),title('原始图片b')
subplot(324),imhist(pic2_gray)
subplot(325),imshow(pic3_gray),title('原始图片c')
subplot(326),imhist(pic3_gray)

%% 混合矩阵 
%将原图像矩阵变为向量
m1=reshape(pic1_gray,[1,length*width]);
m2=reshape(pic2_gray,[1,length*width]);
m3=reshape(pic3_gray,[1,length*width]); 
m=[m1;m2;m3];                                     %将向量合并为一个矩阵
pic=double(m);                                    
A=rand(size(pic,1));                              %随机生成系数矩阵，
mixedpic=A*pic;                                   %混合矩阵生成
mixedpic1=reshape(mixedpic(1,:),[length,width]);
mixedpic2=reshape(mixedpic(2,:),[length,width]);
mixedpic3=reshape(mixedpic(3,:),[length,width]);
MixedPicUnit1=uint8 (round(mixedpic1));
MixedPicUnit2=uint8 (round(mixedpic2));
MixedPicUnit3=uint8 (round(mixedpic3));

%显示混合图像及直方图
figure
subplot(131),imshow(MixedPicUnit1),title('混合图片a');
subplot(132),imshow(MixedPicUnit2),title('混合图片b');
subplot(133),imshow(MixedPicUnit3),title('混合图片c');
mixedpicture=mixedpic;                         

%% 预处理
%标准化
mixedpic_rawmean=zeros(3,1);

for i=1:size(m,1)
    mixedpic_rawmean(i)=mean(mixedpic(i,:));      %计算混合图象每一行的均值
end 

for i=1:size(m,1)
    for j=1:size(mixedpic,2)
        mixedpic(i,j)=mixedpic(i,j)-mixedpic_rawmean(i);%混合矩阵减去对应行的均值
    end
end
%白化
mixedpic_cov=cov(mixedpic');                     %求协方差矩阵
[M,N]=eig(mixedpic_cov);                         %对协方差矩阵进行特征值分解
WhiteningMatrix=inv(sqrt(N))*(M)';               %求白化矩阵
WhiteningMixedPic=WhiteningMatrix*mixedpic;      %求白化后的混合矩阵

%% FastICA

X=WhiteningMixedPic;
[vnum,samplenum]=size(X);
ICNum=vnum;                                      %独立元个数
W=zeros(ICNum,vnum);                           
for r=1:ICNum
    i=1;
    nmax=200;                                    %最大迭代次数
    w=2*(rand(ICNum,1)-0.5);                      %分离向量初始化
    w=w/norm(w);                                 %标准化
    
    while i<=nmax+1
        if i==nmax
            fprintf('\n 第%d分量在%d次迭代内不收敛.',r,nmax);
            break;                               %如果nmax次内不收敛，退出
        end
        w_save=w;                                  %寄存w
        a=1;
        t=X'*w;
        G=t.^3;
        g=3*t.^2;
        w=((1-a)*t'*G*w+a*X*G)/samplenum-mean(g)*w;%迭代公式
        w=w-W*W'*w;                              %正交化公式
        w=w/norm(w);                             %标准化公式
        alpha=abs(abs(w'*w_save)-1);               %收敛精度alpha
        if alpha<1e-9
            W(:,r)=w;                            %如果收敛，保存分离向量，生成分离矩阵
            break;
        end
        i=i+1;
    end
end

%% 图像分离

ICAresults=W'*WhiteningMatrix*mixedpicture;      %图像分离

ICAresults=abs(50*ICAresults);                   %图像增强
                                                 %图像重构
SeparatedPic1=reshape(ICAresults(1,:),[length width]);
SeparatedPic2=reshape(ICAresults(2,:),[length width]);
SeparatedPic3=reshape(ICAresults(3,:),[length width]);
SeparatedPicUnit1=uint8(round(SeparatedPic1));
SeparatedPicUnit2=uint8(round(SeparatedPic2));
SeparatedPicUnit3=uint8(round(SeparatedPic3));

%显示分离后的图像
figure
subplot(321),imshow(SeparatedPicUnit1),title('分离出的图片1')
subplot(322),imhist(SeparatedPicUnit1)
subplot(323),imshow(SeparatedPicUnit2),title('分离出的图片2')
subplot(324),imhist(SeparatedPicUnit2)
subplot(325),imshow(SeparatedPicUnit3),title('分离出的图片3')
subplot(326),imhist(SeparatedPicUnit3)

% 
% 
%差值计算，需人工匹配
% InterPic1=imsubtract(pic1_gray,SeparatedPicUnit1);
% InterPic2=imsubtract(pic2_gray,SeparatedPicUnit2);
% InterPic3=imsubtract(pic3_gray,SeparatedPicUnit3);
% % 显示分离后的图片与原图的差值图以及对应的直方图
% figure
% subplot(321),imshow(InterPic1),title('差值图片1'),subplot(322),imhist(InterPic1),title('差值图片1的直方图');
% subplot(323),imshow(InterPic2),title('差值图片2'),subplot(324),imhist(InterPic2),title('差值图片2的直方图');
% subplot(325),imshow(InterPic3),title('差值图片3'),subplot(326),imhist(InterPic3),title('差值图片3的直方图');



