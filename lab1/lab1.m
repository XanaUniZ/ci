%-------------------------------------------------------------------------
% University of Zaragoza
%
% Author:  M. Peribañez
% Author:  X. Anadon
%-------------------------------------------------------------------------
% SLAM for Karel the robot in 1D
%-------------------------------------------------------------------------

clear all; % varialbes
close all; % figures
randn('state', 1); % always use same random number sequence
rand('state', 1); % always use same random number sequence
format long
%-------------------------------------------------------------------------
% Ex. 1: Read the image 

imageData = imread('IMG_0596.tiff');

% Get the size of the image
[height, width, numChannels] = size(imageData);

imageDataDouble = double(imageData);

% bitDepthPerChannel = 8 * numel(typecast(cast(0, class(imageData)), 'uint8')); 
bitDepthPerChannel = 8; % This is the same as the image is uint8
bitsPerPixel = bitDepthPerChannel * (numChannels);

% Display the results
fprintf('Bits per pixel: %d\n', bitsPerPixel);
fprintf('Image width: %d\n', width);
fprintf('Image height: %d\n', height);
fprintf('Image numChannels: %d\n', numChannels);

% Visualize the image
% figure; imshow(imageData);
%-------------------------------------------------------------------------
% Ex. 2: Linearization
min_val = 1023;
max_val = 15600;
imageDataLin = (imageDataDouble-min_val)/(max_val-min_val);

% Ensuring not out of range
imageDataLin = max(0, min(1, imageDataLin));

% Display the results
fprintf('\nMin: %.4f\n', min(imageDataLin(:)));
fprintf('Max: %.4f\n', max(imageDataLin(:)));

% Optionally, visualize the image
% figure;imshow(imageDataLin);

%-------------------------------------------------------------------------
% Ex. 3: Demosaicing
% Identify the patern
% topLeftSquare = imageDataLin(1:2, 1:2);
% disp('Top-Left 2x2 Square:');
% disp(topLeftSquare);

% demosaic_nn()
% demosaic_bl()

% imgDemNN = demosaic_nn(imageDataLin);
imgDem = demosaic_bl(imageDataLin);
% figure;imshow(imgDem);
% For now use the matlab function
% imageDataLin_uint16 = uint16(imageDataLin * (max_val-min_val) + min_val);
% J = demosaic(imageData,"grbg");
% J = (double(J)-min_val)/(max_val-min_val);
% figure;imshow(J);
% imgDem = demosaic(imageData,"rggb"); % This is the one
% imgDem = (double(imgDem)-min_val)/(max_val-min_val);
% figure;imshow(imgDem);
% J = demosaic(imageData,"bggr");
% J = (double(J)-min_val)/(max_val-min_val);
% figure;imshow(J);
% J = demosaic(imageData,"gbrg");
% J = (double(J)-min_val)/(max_val-min_val);
% figure;imshow(J);

% % Functions Ex. 3
function demImage = demosaic_bl (inImage)
    % Get the size of the input image
    [height, width] = size(inImage);
    
    % Preallocate RGB channels
    redChannel = zeros(height, width);
    greenChannel = zeros(height, width);
    blueChannel = zeros(height, width);

     % Extract RGGB pattern components
    % Red channel (R in RGGB is at [1,1])
    redChannel(1:2:end, 1:2:end) = inImage(1:2:end, 1:2:end);
    
    % Blue channel (B in RGGB is at [2,2])
    blueChannel(2:2:end, 2:2:end) = inImage(2:2:end, 2:2:end);
    
    % Green channel (G in RGGB is at [1,2] and [2,1])
    greenChannel(1:2:end, 2:2:end) = inImage(1:2:end, 2:2:end);
    greenChannel(2:2:end, 1:2:end) = inImage(2:2:end, 1:2:end);

    % Perform bilinear interpolation
    % Interpolate Red channel by using convolution
    redChannel = convolve(redChannel);
    % Interpolate Green channel by using convolution
    greenChannel = convolveG(greenChannel);
    % Interpolate Blue channel by using convolution
    blueChannel = convolve(blueChannel);
    
    % Put them together
    demImage = cat(3, redChannel, greenChannel, blueChannel);

end

function channel = convolve(channel_in)
    % Create a mask of known values
    mask = channel_in > 0;
    
    % Interpolate missing values using convolution
    kernel = [1, 1, 1; 1, 1, 1; 1, 1, 1]; % Bilinear weights
    % kernel = kernel / sum(kernel(:));    % Normalize the kernel
    
    % Apply convolution to the image and the mask
    interpolated = conv2(channel_in, kernel, 'same');
    weight = conv2(double(mask), kernel, 'same');
    
    % Combine interpolated values with known values
    channel = interpolated ./ weight; % Normalize interpolation by weights
    % channel(mask) = channel_in(mask);   
end

function channel = convolveG(channel_in)
    % Create a mask of known values
    mask = channel_in > 0;
    
    % Interpolate missing values using convolution
    kernel = [0, 1, 0; 1, 1, 1; 0, 1, 0]; % Bilinear weights
    kernel = kernel / sum(kernel(:));    % Normalize the kernel
    
    % Apply convolution to the image and the mask
    interpolated = conv2(channel_in, kernel, 'same');
    weight = conv2(double(mask), kernel, 'same');
    
    % Combine interpolated values with known values
    channel = interpolated ./ weight; % Normalize interpolation by weights
    % channel(mask) = channel_in(mask);   
end

%-------------------------------------------------------------------------
% Ex. 4: White balancing
% Gray World Balancing
imgGWbal = gw_balancing(imgDem);
% figure;imshow(imgGWbal);

% Gray World Balancing
imgWWbal = ww_balancing(imgDem);
% figure;imshow(imgWWbal);

% Manual Balancing

imgBalanced = imgGWbal;

% Functions Ex. 4
function balancedIm = gw_balancing (inImage)
    rAvg = mean(inImage(:,:,1));
    gAvg = mean(inImage(:,:,2));
    bAvg = mean(inImage(:,:,3));
    balancer = [gAvg/rAvg, 1, gAvg/bAvg];
    balancedIm = inImage .* reshape(balancer, 1, 1, []);
end

function balancedIm = ww_balancing (inImage)
    rMax = max(inImage(:,:,1));
    gMax = max(inImage(:,:,2));
    bMax = max(inImage(:,:,3));
    balancer = [gMax/rMax, 1, gMax/bMax];
    balancedIm = inImage .* reshape(balancer, 1, 1, []);
end
%-------------------------------------------------------------------------
% Ex. 5: Denoising
imgDenoised = denoise(imgBalanced, @denoiseGaussian);
figure;imshow(imgDenoised);

% Functions Ex. 5
function denoised = denoise(inImage, denoiseFunc) 
    kernelSize = 3;
    % Denoise each channel
    redChannel = denoiseFunc(inImage(:,:,1), kernelSize);
    greenChannel = denoiseFunc(inImage(:,:,2), kernelSize);
    blueChannel = denoiseFunc(inImage(:,:,3), kernelSize);

    % Put them together
    denoised = cat(3, redChannel, greenChannel, blueChannel);
    
end

function denoised = denoiseMean(channel_in, kSize)    
    % kernel = [1, 1, 1; 1, 1, 1; 1, 1, 1]; % weights
    kernel = ones(kSize);
    kernel = kernel / (kSize*kSize);    % Normalize the kernel
    
    % Apply convolution to the image and the mask
    denoised = conv2(channel_in, kernel, 'same'); 
end

function denoised = denoiseMedian(channel_in, kSize)    
    % Interpolate missing values using convolution
    kernel = [1, 1, 1; 1, 1, 1; 1, 1, 1]; % Bilinear weights
    kernel = kernel / sum(kernel(:));    % Normalize the kernel
    
    % Apply convolution to the image and the mask
    denoised = conv2(channel_in, kernel, 'same'); 
end

function denoised = denoiseGaussian(channel_in, kSize)
    % Mean and std
    std = 1.;

    % Create a grid with X and Y
    [X, Y] = meshgrid(-kSize:kSize, -kSize:kSize);

    % Compute Gaussian function for the kernel
    kernel = exp(-(X.^2 + Y.^2) / (2 * std^2)) / (2 * pi * std^2);
    kernel = kernel / sum(kernel(:));
    
    % Apply convolution to the image and the mask
    denoised = conv2(channel_in, kernel, 'same'); 
end

%-------------------------------------------------------------------------
% Ex. 6: Color balance

% Functions Ex. 6

%-------------------------------------------------------------------------
% Ex. 7: Tone reproduction

% Functions Ex. 7

%-------------------------------------------------------------------------
% Ex. 8: Compression

% Functions Ex. 8
