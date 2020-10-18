%% 
clc;% clean command window
clear;% clean workspace
%% Loading and viewing image
I = imread('rosetta.jpg');
imshow(I)

%% Check whether the image is rgb or grayscale
% If image is rgb,  convert it to grayscale
% Else, it remains as it is.
[m,n,d]=size(I);
if d > 1
    fprintf("RGB Image\n")
    y = rgb2gray(I);
    imshow(y)
end
if d == 1
    fprintf("Already Grayscale Image\n")
end
%% Convert the pixel values to double values between 0 and 1
y = double(y)/255;
% Task 1 
%%
p=0.5;
rng('default')
ind = randperm(size(y,1)*size(y,2));
ind_o = ind(1:round(p*length(ind)));
ind_m = ind(round(p*length(ind))+1:length(ind));
n_tot = length(ind);
%% 
L = 1;
xx = linspace(0,L,size(y,1));
yy = linspace(0,L,size(y,2));
[XX, YY] = meshgrid(xx,yy);
loc = [XX(:) YY(:)];
%pdist(loc);% Compute the Euclidean distance
%% Parameter estimation by OLS
n_obs = 10000;
ind_nobs = ind_o(1:n_obs); % 10000 (random) first observed pixels
B1 = ones(n_tot,1);% Basis functions
%B2 = loc(:,1);
%ind_o=linspace(1,n_obs,n_obs);
B=B1(ind_nobs); % B2(ind_nobs)];
%%
y=reshape(y,n_tot,1);
%%
Y_nobs=y(ind_nobs);
beta_hat = (B'*B)\(B'*Y_nobs)
mu_OLS=B*beta_hat;
e = Y_nobs - mu_OLS;
%% Plot the binned variogram estimate as well as the estimated Matern variogram
l=loc(ind_nobs,:);
N=100;% No of bins
r1=emp_variogram(l,e,N);
fixed=struct('nu',1);
pars=cov_ls_est(Y_nobs,'matern',r1,fixed)
%%
std(e)
%%
plot(r1.h,r1.variogram)
hold on
sigma = pars(1).sigma;
kappa = pars(1).kappa;% r=sqrt(8*nu)/kappa~0.5
nu = pars(1).nu;% Given in problem
sigma_e=pars(1).sigma_e;
r2=matern_variogram(r1.h,sigma,kappa,nu,sigma_e);
plot1=plot(r1.h,r2)
hold off
title('Binned variogram estimate & estimated Matérn variogram plot')
saveas(plot1,'img1(p=0,5)','png');
%% Parameter estimation by GLS based on observed first 10000 pixels
D = squareform(pdist(l));% Squareform returns a symmetric matrix where Z(i,j) corresponds to the pairwise distance between observations i and j
cov_GLS = matern_covariance(D,sigma,kappa,nu);
B=B1(ind_nobs);
beta_GLS = (B'*inv(cov_GLS)*B)\(B'*inv(cov_GLS)*Y_nobs)
mu_GLS=B*beta_GLS;
%% Simulation of GMRFs 
q = kappa^4*[0 0 0 0 0;0 0 0 0 0;0 0 1 0 0;0 0 0 0 0;0 0 0 0 0]+2*kappa^2*[0 0 0 0 0;0 0 -1 0 0;0 -1 4 -1 0;0 0 -1 0 0;0 0 0 0 0] + [0 0 1 0 0;0 2 -8 2 0;1 -8 20 -8 1;0 2 -8 2 0;0 0 1 0 0];
tau = 2*pi/sigma^2;
Q = tau*stencil2prec([m,n],q);%[m,n] is the size of the image
%% based on 10000 first missing
Qpo2 = Q(ind_m,ind_o(1:n_obs));
Qp2 = Q(ind_m,ind_m);
%R2 = chol(Qp2); % cholesky factor 
%% 
I2=Qp2\(Qpo2*(Y_nobs-mu_OLS));
E_m2 = beta_GLS-I2; % mu_ols gives better result?
plot(ind_m,E_m2)
hold on
plot(ind_m,y(ind_m))
hold off
%%
E_all = zeros(length(ind),1);
E_all(ind_o)=y(ind_o); 
E_all(ind_m)= E_m2;
%%
E_a = zeros(length(ind),1);
E_a(ind_o)=y(ind_o); 
E_a(ind_m)= zeros(length(ind_m),1);
y2=reshape(E_a,[m,n])
imagesc(y2)
colormap gray
%%
y_reconstructed = reshape(E_all,[m,n]);
subplot(1,2,1)
%imshow('titan.jpg')
imagesc(y2)
title('Previous image')
plot2=subplot(1,2,2)
imagesc(y_reconstructed)
title('Reconstructed image')
colormap gray
suptitle('Reconstruction using 50% missing pixel values')
saveas(plot2,'img2(p=0,5)','png');