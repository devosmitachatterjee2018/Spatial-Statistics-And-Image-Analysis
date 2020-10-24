%% Project 2 without noise 
image = load('permeability.mat');
image = image.Y;
J = image;
x = reshape(J,[],size(J,3),1);
x = double(x);

%% Original image plot 
imagesc(image);
axis image
set(gca,'xtick',[],'ytick',[])
colormap gray
colorbar
set(gcf, 'Color', 'w');
export_fig original_image.png


%% K-means 
K = 2; % K = 2 since we want 2 segments; high and low permeability areas
maxiter = 20;
verbose = 1;
cl_kmeans = normmix_kmeans(x,K,maxiter,verbose);
kmeans = reshape(cl_kmeans,[],size(image,2),1);

imagesc(kmeans)
axis image
set(gca,'xtick',[],'ytick',[])
colormap gray
colorbar
set(gcf, 'Color', 'w');
export_fig P2_kmeans.png

%% GMM
K = 2;
Niter = 10;
step0 = 1;
plotflag = 0;
[pars,traj] = normmix_sgd(x,K,Niter,step0,plotflag);
[cl_gmm,p] = normmix_classify(x,pars);
GMM = reshape(cl_gmm,[],size(image,2),1);
%%
imagesc(GMM);
axis image
set(gca,'xtick',[],'ytick',[])
colormap gray
colorbar
set(gcf, 'Color', 'w');
export_fig P2_GMM.png

%% MRF 
N = [0 1 0;1 0 1;0 1 0]; % Neighbourhood structure
K = 2; 
opts = struct('plot',0,'N',N,'common_beta',1);
[theta,alpha,beta,cl_mrf,p] = mrf_sgd(x,K,opts);
MRF = reshape(cl_mrf,[],size(image,2),1);
%%
imagesc(MRF)
axis image
set(gca,'xtick',[],'ytick',[])
colormap gray
colorbar
set(gcf, 'Color', 'w');
export_fig P2_MRF.png

%% Comparison between results
pixels = 13200;

kmeans_GMM = sum(cl_kmeans == cl_gmm); % 13093
kmeans_GMM_percent = kmeans_GMM/pixels; % 99.19%

kmeans_MRF = sum(cl_kmeans == cl_mrf); % 13087
kmeans_MRF_percent = kmeans_MRF/pixels; % 99.14%

GMM_MRF = sum(cl_gmm == cl_mrf); % 13178
GMM_MRF_percent = GMM_MRF/pixels; % 99.83%