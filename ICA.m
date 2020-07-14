clear
clc
close all
%% ͼƬ����׼��
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
% ԭʼͼ��ֱ��ͼ
figure
subplot(321),imshow(pic1_gray),title('ԭʼͼƬa')
subplot(322),imhist(pic1_gray)
subplot(323),imshow(pic2_gray),title('ԭʼͼƬb')
subplot(324),imhist(pic2_gray)
subplot(325),imshow(pic3_gray),title('ԭʼͼƬc')
subplot(326),imhist(pic3_gray)

%% ��Ͼ��� 
%��ԭͼ������Ϊ����
m1=reshape(pic1_gray,[1,length*width]);
m2=reshape(pic2_gray,[1,length*width]);
m3=reshape(pic3_gray,[1,length*width]); 
m=[m1;m2;m3];                                     %�������ϲ�Ϊһ������
pic=double(m);                                    
A=rand(size(pic,1));                              %�������ϵ������
mixedpic=A*pic;                                   %��Ͼ�������
mixedpic1=reshape(mixedpic(1,:),[length,width]);
mixedpic2=reshape(mixedpic(2,:),[length,width]);
mixedpic3=reshape(mixedpic(3,:),[length,width]);
MixedPicUnit1=uint8 (round(mixedpic1));
MixedPicUnit2=uint8 (round(mixedpic2));
MixedPicUnit3=uint8 (round(mixedpic3));

%��ʾ���ͼ��ֱ��ͼ
figure
subplot(131),imshow(MixedPicUnit1),title('���ͼƬa');
subplot(132),imshow(MixedPicUnit2),title('���ͼƬb');
subplot(133),imshow(MixedPicUnit3),title('���ͼƬc');
mixedpicture=mixedpic;                         

%% Ԥ����
%��׼��
mixedpic_rawmean=zeros(3,1);

for i=1:size(m,1)
    mixedpic_rawmean(i)=mean(mixedpic(i,:));      %������ͼ��ÿһ�еľ�ֵ
end 

for i=1:size(m,1)
    for j=1:size(mixedpic,2)
        mixedpic(i,j)=mixedpic(i,j)-mixedpic_rawmean(i);%��Ͼ����ȥ��Ӧ�еľ�ֵ
    end
end
%�׻�
mixedpic_cov=cov(mixedpic');                     %��Э�������
[M,N]=eig(mixedpic_cov);                         %��Э��������������ֵ�ֽ�
WhiteningMatrix=inv(sqrt(N))*(M)';               %��׻�����
WhiteningMixedPic=WhiteningMatrix*mixedpic;      %��׻���Ļ�Ͼ���

%% FastICA

X=WhiteningMixedPic;
[vnum,samplenum]=size(X);
ICNum=vnum;                                      %����Ԫ����
W=zeros(ICNum,vnum);                           
for r=1:ICNum
    i=1;
    nmax=200;                                    %����������
    w=2*(rand(ICNum,1)-0.5);                      %����������ʼ��
    w=w/norm(w);                                 %��׼��
    
    while i<=nmax+1
        if i==nmax
            fprintf('\n ��%d������%d�ε����ڲ�����.',r,nmax);
            break;                               %���nmax���ڲ��������˳�
        end
        w_save=w;                                  %�Ĵ�w
        a=1;
        t=X'*w;
        G=t.^3;
        g=3*t.^2;
        w=((1-a)*t'*G*w+a*X*G)/samplenum-mean(g)*w;%������ʽ
        w=w-W*W'*w;                              %��������ʽ
        w=w/norm(w);                             %��׼����ʽ
        alpha=abs(abs(w'*w_save)-1);               %��������alpha
        if alpha<1e-9
            W(:,r)=w;                            %�������������������������ɷ������
            break;
        end
        i=i+1;
    end
end

%% ͼ�����

ICAresults=W'*WhiteningMatrix*mixedpicture;      %ͼ�����

ICAresults=abs(50*ICAresults);                   %ͼ����ǿ
                                                 %ͼ���ع�
SeparatedPic1=reshape(ICAresults(1,:),[length width]);
SeparatedPic2=reshape(ICAresults(2,:),[length width]);
SeparatedPic3=reshape(ICAresults(3,:),[length width]);
SeparatedPicUnit1=uint8(round(SeparatedPic1));
SeparatedPicUnit2=uint8(round(SeparatedPic2));
SeparatedPicUnit3=uint8(round(SeparatedPic3));

%��ʾ������ͼ��
figure
subplot(321),imshow(SeparatedPicUnit1),title('�������ͼƬ1')
subplot(322),imhist(SeparatedPicUnit1)
subplot(323),imshow(SeparatedPicUnit2),title('�������ͼƬ2')
subplot(324),imhist(SeparatedPicUnit2)
subplot(325),imshow(SeparatedPicUnit3),title('�������ͼƬ3')
subplot(326),imhist(SeparatedPicUnit3)

% 
% 
%��ֵ���㣬���˹�ƥ��
% InterPic1=imsubtract(pic1_gray,SeparatedPicUnit1);
% InterPic2=imsubtract(pic2_gray,SeparatedPicUnit2);
% InterPic3=imsubtract(pic3_gray,SeparatedPicUnit3);
% % ��ʾ������ͼƬ��ԭͼ�Ĳ�ֵͼ�Լ���Ӧ��ֱ��ͼ
% figure
% subplot(321),imshow(InterPic1),title('��ֵͼƬ1'),subplot(322),imhist(InterPic1),title('��ֵͼƬ1��ֱ��ͼ');
% subplot(323),imshow(InterPic2),title('��ֵͼƬ2'),subplot(324),imhist(InterPic2),title('��ֵͼƬ2��ֱ��ͼ');
% subplot(325),imshow(InterPic3),title('��ֵͼƬ3'),subplot(326),imhist(InterPic3),title('��ֵͼƬ3��ֱ��ͼ');



