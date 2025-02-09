% Read data
aperture = imread('apertures/circular.bmp');
image   = imread('images/penguins.jpg');  
% (Do not extract a single channel so that we preserve color)

% Noise level (Gaussian noise)
sigma = 0.005;

% Blur size
blurSize = 7;

% Convert image to double
f0 = im2double(image);
[height, width, channels] = size(f0);

% Prior matrix: 1/f law
A_star = eMakePrior(height, width) + 1e-8;
C = sigma.^2 * height * width ./ A_star;

% Normalization (using fspecial to compute a disk filter)
temp = fspecial('disk', blurSize);
flow = max(temp(:));

% Calculate effective PSF
k1 = im2double(imresize(aperture, [2*blurSize + 1, 2*blurSize + 1], 'nearest'));
k1 = k1 * (flow / max(k1(:)));
% Normalize PSF so that it sums to one
k1 = k1 / sum(k1(:));

%% Apply blur to each color channel independently
f1 = zeros(size(f0));
for c = 1:channels
    % The zDefocused function is assumed to work on 2D images.
    f1(:,:,c) = zDefocused(f0(:,:,c), k1, sigma, 0);
end

%% Recover Original
% Option 1: Lucy–Richardson Deconvolution (deconvlucy)
num_iter = 30;  % you can experiment with different numbers of iterations
f0_hat = zeros(size(f0));
for c = 1:channels
    f0_hat(:,:,c) = deconvlucy(f1(:,:,c), k1, num_iter);
end

% %% Option 2: Wiener Deconvolution (deconvwnr)
% NSR = sigma^2;  
% f0_hat = zeros(size(f0));
% for c = 1:channels
%     f0_hat(:,:,c) = deconvwnr(f1(:,:,c), k1, NSR);
% end

%% Display results
figure;

subplot_tight(1, 3, 1, 0.0, false)
imshow(f0);
title('Focused');

subplot_tight(1, 3, 2, 0.0, false)
imshow(f1);
title('Defocused');

subplot_tight(1, 3, 3, 0.0, false)
imshow(f0_hat);
title('Recovered');