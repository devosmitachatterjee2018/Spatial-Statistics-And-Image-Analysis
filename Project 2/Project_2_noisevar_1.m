%% Now adding noise to data 
% Since this data set is simulated, there is no noise in it. However, one might expect real measurements
% of permeability to be much more noisy. Generate a “measured” data set by adding independent
% mean-zero Gaussian noise, with standard deviation , to each pixel in the image.
image = load('permeability.mat');
image = image.Y;
%%  adding noise var = 1 mean = 0
noise_sigma = 1;
noise = noise_sigma * randn(size(image));
J_1 = image + noise;

x = reshape(J_1,[],size(J_1,3),1);
x = double(x);

%% plot of image with noise 
imagesc(J_1);
axis image
set(gca,'xtick',[],'ytick',[])
colormap gray
colorbar
set(gcf, 'Color', 'w');
export_fig P2_noisy_var1.png

%% K-means 
K = 2; % K = 2 since we want 2 segments; high and low permeability areas
maxiter = 20;
verbose = 1;
cl_kmeans = normmix_kmeans(x,K,maxiter,verbose);
kmeans = reshape(cl_kmeans,[],size(image,2),1);

%%
imagesc(kmeans)
axis image
set(gca,'xtick',[],'ytick',[])
colormap gray
colorbar
set(gcf, 'Color', 'w');
export_fig P2_kmeans_var1.png

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
export_fig P2_GMM_var1.png

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
export_fig P2_MRF_var1.png

%% Comparison between results
pixels = 13200;

kmeans_GMM = sum(cl_kmeans == cl_gmm); % 13085
kmeans_GMM_percent = kmeans_GMM/pixels; % 99.13%

kmeans_MRF = sum(cl_kmeans == cl_mrf); % 12982
kmeans_MRF_percent = kmeans_MRF/pixels; % 98.35%

GMM_MRF = sum(cl_gmm == cl_mrf); % 13025
GMM_MRF_percent = GMM_MRF/pixels; % 98.67%
