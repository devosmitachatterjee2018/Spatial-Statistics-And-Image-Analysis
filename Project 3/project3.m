clc
%clear all
tms016path
%% Loading images
%image1=mat2gray(imread('im1.tif')); %test
ImDC11=mat2gray(imread('HA1200_0.65_20keV_DC004_01_001_pag_1613.tif'));
ImDC21=mat2gray(imread('HA1200_0.65_20keV_DC004_03_001_pag_1701.tif')); % Needs fixed filters
ImRC11=mat2gray(imread('HA1200_0.65_20keV_RC002_01_001_pag_1729.tif'));
ImTS11=mat2gray(imread('HA1200_0.65_20keV_TSG003_01_001_pag_1584.tif'));

%% Set image for analysis
I=ImDC11;
[m,n,d]=size(I);

%%
figure(23)
subplot(2,2,1)
imagesc(ImDC11)
title('Direct Compression tablet 1')
axis image
subplot(2,2,2)
imagesc(ImDC11)
title('Direct Compression tablet 2')
axis image
subplot(2,2,3)
imagesc(ImRC11)
title('Rolling Compression tablet 1')
axis image
subplot(2,2,4)
imagesc(ImTS11)
title('Twin-Screw Granulation tablet 1')
axis image
%% 
figure(1)
ax1=subplot(1,2,1);
imagesc(I)
colormap(ax1,gray)
title('Original grayscale')
axis image
ax2=subplot(1,2,2);
imagesc(I)
colormap(ax2,jet)
title('Original intense colours')
axis image
savefig('ImDC11');
%% Edge detection
figure(2)
[~,threshold] = edge(I,'sobel');
fudgeFactorless = 0.5; 
fudgeFactorhigh = 0.05;
% finding initial borders/edges
BWsless = edge(I,'sobel',threshold * fudgeFactorless);
BWshigh = edge(I,'sobel',threshold * fudgeFactorhigh);
subplot(2,2,1)
imshow(BWsless)
title('Binary Gradient Mask')

se90 = strel('line',3,90);
se0 = strel('line',3,0);
% diluting image to get complete borders/edges
BWsdilless = imdilate(BWsless,[se90 se0]);
BWsdilless = imdilate(BWsdilless,[se90 se0]);
BWsdilless = imdilate(BWsdilless,[se90 se0]);
BWsdilhigh = imdilate(BWshigh,[se90 se0]);
BWsdilhigh = imdilate(BWsdilhigh,[se90 se0]);
BWsdilhigh = imdilate(BWsdilhigh,[se90 se0]);
subplot(2,2,2)
imshow(BWsdilless)
title('Dilated Gradient Mask')

% filling the holes in the object
BWdfillless = imfill(BWsdilless,'holes');
BWdfillhigh = imfill(BWsdilhigh,'holes');

%imshow(BWdfillless)
%title('Binary Image with Filled Holes')

% removing objects on border. Bad Idea since our object is on the border!
%BWnobordless = imclearborder(BWdfillless,4);
%BWnobordhigh = imclearborder(BWdfillhigh,4);

%imshow(BWnobordless)
%title('Cleared Border Image')

% smoothing the border with diamond 
seD = strel('diamond',1);
%BWfinalless = imerode(BWnobordless,seD);
BWfinalless = imerode(BWdfillless,seD);
%BWfinalhigh = imerode(BWnobordhigh,seD);
BWfinalhigh = imerode(BWdfillhigh,seD);
subplot(2,2,3)
imshow(BWfinalless)
title('Low Background removal filter');
subplot(2,2,4)
imshow(BWfinalhigh)
title('High Background removal filter');
savefig('FiltersImDC11');
%% results from "filters"
figure(3)
subplot(1,3,1)
imshow(I)
title('Original image')
Iless=labeloverlay(I,BWfinalless); % not including outer layer of tablet
Ihigh=labeloverlay(I,BWfinalhigh); % including outer layer of tablet
subplot(1,3,2)
imshow(Iless)
title('Low Mask Over Original Image')
subplot(1,3,3)
imshow(Ihigh)
title('High Mask Over Original Image')
savefig('FilterOnImDC11');
%% Removing background and stacking (1-dimension)
ind_tot=[1:m*n*d];
Ied=(I+0.2).*BWfinalless; % edited Image to perform segmentation on. I made this up! Does it work?
IedH=(I+0.2).*BWfinalhigh; % edited Image to perform segmentation on. I made this up! Does it work?
xStacked = reshape(Ied,[],size(Ied,3),1);
xStackedH = reshape(IedH,[],size(IedH,3),1);
xd1=double(xStacked);
xd1H=double(xStackedH);

ind_obj=find(xd1>0); % removing background
ind_back=find(xd1==0);
xd1ed=xd1(ind_obj);

ind_objH=find(xd1H>0); % removing background
ind_backH=find(xd1H==0);
xd1edH=xd1H(ind_objH);

%% plotting historgrams
figure(24)
subplot(2,2,1)
histogram(I)
title('Original image')
subplot(2,2,2)
histogram(xd1)
title('image with separated/distinguished background')
subplot(2,2,3)
histogram(xd1ed)
title({'Bakground removed with low filter,','leaving less backgroun'})
subplot(2,2,4)
histogram(xd1edH)
title({'Background removed with high filter,','leaving more background'})
savefig('HistogramImDC11');

%% GMM settings for both filters K=3
K=3; % when changing to K, change resultKrL and resultKrH accordingly
Niter=20; 
step0=1;
%% GMM Segmentation for low filter
% Be careful when running this! It takes time, especially for large K and Niter
figure(4)
plotflag=1;
[pars,traj]=normmix_sgd(xd1ed,K,Niter,step0,plotflag);
[cl,p]=normmix_classify(xd1ed,pars);

%% 
cled = zeros(m*n,1);
cled(ind_obj) = cl;
cled(ind_back)= 0; % unnecessary since the base is 0
result3rL=reshape(cled,[],size(I,2),1);
%% Visualising result low filter
figure(6)
subplot(2,3,1)
imshow(result3rL,lines)
title('Combined GMM K=3 for low filter') %jet?
axis image
CLtot=length(ind_obj);

subplot(2,3,2)
imagesc(result3rL==1)
title('GMM K=3 segment 1')
axis image
CL1=sum(result3rL(:)==1);

subplot(2,3,3)
imagesc(result3rL==2)
title('GMM K=3 segment 2')
axis image
CL2=sum(result3rL(:)==2);

subplot(2,3,4)
imagesc(result3rL==3)
title('GMM K=3 segment 3')
axis image
CL3=sum(result3rL(:)==3);

subplot(2,3,5)
imagesc(result3rL==4)
title('GMM K=3 segment 4')
axis image
CL4=sum(result3rL(:)==4);

subplot(2,3,6)
imagesc(result3rL==5)
title('GMM K=3 segment 5')
axis image
CL5=sum(result3rL(:)==5);

% computing component shares:
pL3=[CL1/CLtot CL2/CLtot CL3/CLtot CL4/CLtot CL5/CLtot];

savefig('GMM3LImDC11');
%% GMM Segmentation for high filter
% Be careful when running this! It takes time, especially for large K and Niter
figure(7)
plotflagH=1;
[parsH,trajH]=normmix_sgd(xd1ed,K,Niter,step0,plotflagH);
[clH,PH]=normmix_classify(xd1edH,parsH);
%% 
cledH = zeros(m*n,d);
cledH(ind_objH) = clH;
cledH(ind_backH)= 0;
result3rH=reshape(cledH,[],size(I,2),1);
%% Visualising result high filter
figure(10)
subplot(2,3,1)
imshow(result3rH,lines)
title('Combined GMM K=3 for high filter') %jet?
axis image
CHtot=length(ind_objH);

subplot(2,3,2)
imagesc(result3rH==1)
title('GMM K=3 segment 1')
axis image
CH1=sum(result3rH(:)==1);

subplot(2,3,3)
imagesc(result3rH==2)
title('GMM K=3 segment 2')
axis image
CH2=sum(result3rH(:)==2);

subplot(2,3,4)
imagesc(result3rH==3)
title('GMM K=3 segment 3')
axis image
CH3=sum(result3rH(:)==3);

subplot(2,3,5)
imagesc(result3rH==4)
title('GMM K=3 segment 4')
axis image
CH4=sum(result3rH(:)==4);

subplot(2,3,6)
imagesc(result3rH==5)
title('GMM K=3 segment 5')
axis image
CH5=sum(result3rH(:)==5);
% computing component shares:
pH3=[CH1/CHtot CH2/CHtot CH3/CHtot CH4/CHtot CH5/CHtot];

savefig('GMM3HImDC11');







%% GMM settings for both filters K=4
K=4; % when changing to K, change resultKrL and resultKrH accordingly
Niter=20; 
step0=1;
%% GMM Segmentation for low filter
% Be careful when running this! It takes time, especially for large K and Niter
figure(11)
plotflag=1;
[pars,traj]=normmix_sgd(xd1ed,K,Niter,step0,plotflag);
[cl,p]=normmix_classify(xd1ed,pars);

%% 
cled = zeros(m*n,1);
cled(ind_obj) = cl;
cled(ind_back)= 0; % unnecessary since the base is 0
result4rL=reshape(cled,[],size(I,2),1);
%% Visualising result low filter
figure(13)
subplot(2,3,1)
imshow(result4rL,lines)
title('Combined GMM K=4 for low filter') %jet?
axis image
CLtot=length(ind_obj);

subplot(2,3,2)
imagesc(result4rL==1)
title('GMM K=4 segment 1')
axis image
CL1=sum(result4rL(:)==1);

subplot(2,3,3)
imagesc(result4rL==2)
title('GMM K=4 segment 2')
axis image
CL2=sum(result4rL(:)==2);

subplot(2,3,4)
imagesc(result4rL==3)
title('GMM K=4 segment 3')
axis image
CL3=sum(result4rL(:)==3);

subplot(2,3,5)
imagesc(result4rL==4)
title('GMM K=4 segment 4')
axis image
CL4=sum(result4rL(:)==4);

subplot(2,3,6)
imagesc(result4rL==5)
title('GMM K=4 segment 5')
axis image
CL5=sum(result4rL(:)==5);

% computing component shares:
pL4=[CL1/CLtot CL2/CLtot CL3/CLtot CL4/CLtot CL5/CLtot];

savefig('GMM4LImDC11');
%% GMM Segmentation for high filter
% Be careful when running this! It takes time, especially for large K and Niter
figure(14)
plotflagH=1;
[parsH,trajH]=normmix_sgd(xd1ed,K,Niter,step0,plotflagH);
[clH,PH]=normmix_classify(xd1edH,parsH);
%% 
cledH = zeros(m*n,d);
cledH(ind_objH) = clH;
cledH(ind_backH)= 0;
result4rH=reshape(cledH,[],size(I,2),1);
%% Visualising result high filter
figure(16)
subplot(2,3,1)
imshow(result4rH,lines)
title('Combined GMM K=4 for high filter') %jet?
axis image
CHtot=length(ind_objH);

subplot(2,3,2)
imagesc(result4rH==1)
title('GMM K=4 segment 1')
axis image
CH1=sum(result4rH(:)==1);

subplot(2,3,3)
imagesc(result4rH==2)
title('GMM K=4 segment 2')
axis image
CH2=sum(result4rH(:)==2);

subplot(2,3,4)
imagesc(result4rH==3)
title('GMM K=4 segment 3')
axis image
CH3=sum(result4rH(:)==3);

subplot(2,3,5)
imagesc(result4rH==4)
title('GMM K=4 segment 4')
axis image
CH4=sum(result4rH(:)==4);

subplot(2,3,6)
imagesc(result4rH==5)
title('GMM K=4 segment 5')
axis image
CH5=sum(result4rH(:)==5);
% computing component shares:
pH4=[CH1/CHtot CH2/CHtot CH3/CHtot CH4/CHtot CH5/CHtot];

savefig('GMM4HImDC11');






%% GMM settings for both filters K=5
K=5; % when changing to K, change resultKrL and resultKrH accordingly
Niter=20; 
step0=1;
%% GMM Segmentation for low filter
% Be careful when running this! It takes time, especially for large K and Niter
figure(17)
plotflag=1;
[pars,traj]=normmix_sgd(xd1ed,K,Niter,step0,plotflag);
[cl,p]=normmix_classify(xd1ed,pars);

%% 
cled = zeros(m*n,1);
cled(ind_obj) = cl;
cled(ind_back)= 0; % unnecessary since the base is 0
result5rL=reshape(cled,[],size(I,2),1);
%% Visualising result low filter
figure(19)
subplot(2,3,1)
imshow(result5rL,lines)
title('Combined GMM K=5 for low filter') %jet?
axis image
CLtot=length(ind_obj);

subplot(2,3,2)
imagesc(result5rL==1)
title('GMM K=5 segment 1')
axis image
CL1=sum(result5rL(:)==1);

subplot(2,3,3)
imagesc(result5rL==2)
title('GMM K=5 segment 2')
axis image
CL2=sum(result5rL(:)==2);

subplot(2,3,4)
imagesc(result5rL==3)
title('GMM K=5 segment 3')
axis image
CL3=sum(result5rL(:)==3);

subplot(2,3,5)
imagesc(result5rL==4)
title('GMM K=5 segment 4')
axis image
CL4=sum(result5rL(:)==4);

subplot(2,3,6)
imagesc(result5rL==5)
title('GMM K=5 segment 5')
axis image
CL5=sum(result5rL(:)==5);

% computing component shares:
pL5=[CL1/CLtot CL2/CLtot CL3/CLtot CL4/CLtot CL5/CLtot];

savefig('GMM5LImDC11');
%% GMM Segmentation for high filter
% Be careful when running this! It takes time, especially for large K and Niter
figure(20)
plotflagH=1;
[parsH,trajH]=normmix_sgd(xd1ed,K,Niter,step0,plotflagH);
[clH,PH]=normmix_classify(xd1edH,parsH);
%% 
cledH = zeros(m*n,d);
cledH(ind_objH) = clH;
cledH(ind_backH)= 0;
result5rH=reshape(cledH,[],size(I,2),1);
%% Visualising result high filter
figure(22)
subplot(2,3,1)
imshow(result5rH,lines)
title('Combined GMM K=5 for high filter') %jet?
axis image
CHtot=length(ind_objH);

subplot(2,3,2)
imagesc(result5rH==1)
title('GMM K=5 segment 1')
axis image
CH1=sum(result5rH(:)==1);

subplot(2,3,3)
imagesc(result5rH==2)
title('GMM K=5 segment 2')
axis image
CH2=sum(result5rH(:)==2);

subplot(2,3,4)
imagesc(result5rH==3)
title('GMM K=5 segment 3')
axis image
CH3=sum(result5rH(:)==3);

subplot(2,3,5)
imagesc(result5rH==4)
title('GMM K=5 segment 4')
axis image
CH4=sum(result5rH(:)==4);

subplot(2,3,6)
imagesc(result5rH==5)
title('GMM K=5 segment 5')
axis image
CH5=sum(result5rH(:)==5);
% computing component shares:
pH5=[CH1/CHtot CH2/CHtot CH3/CHtot CH4/CHtot CH5/CHtot];

savefig('GMM5HImDC11');