%-------------------------------------------------------------------------
% University of Zaragoza
%
% Author:  M. Peribañez
% Author:  X. Anadon
%-------------------------------------------------------------------------

clear all; % varialbes
close all; % figures
randn('state', 1); % always use same random number sequence
rand('state', 1); % always use same random number sequence
format long
%-------------------------------------------------------------------------
% Ex. 1: Read the image 

imageData = imread('IMG_0596.tiff');
% imageData = imread('IMG_0998.tiff');
% figure; imshow(imageData);

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
image = (imageDataDouble-min_val)/(max_val-min_val);

% Ensuring not out of range
image = max(0, min(1, image));

% Display the results
fprintf('\nMin: %.4f\n', min(image(:)));
fprintf('Max: %.4f\n', max(image(:)));

% Optionally, visualize the image
% figure;imshow(image);

%-------------------------------------------------------------------------
%% Ex. 3: Demosaicing
% Identify the patern
% topLeftSquare = imageDataLin(1:2, 1:2);
% disp('Top-Left 2x2 Square:');
% disp(topLeftSquare);

% demosaic_nn()
% demosaic_bl()

image = my_demosaic(image, @convolveBL, @convolveBLG);
% figure;imshow(imgDem);

% imgDem = my_demosaic(imageDataLin, @convolveNN_RB, @convolveNN_G);
% figure;imshow(imgDem);
% topLeftSquare = imgDem(1:2, 1:2);
% disp('Top-Left 2x2 Square:');
% disp(topLeftSquare);

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
function demImage = my_demosaic (inImage, convolve_RB, convolve_G)
    % Get the size of the input image
    [height, width] = size(inImage);
    
    % Preallocate RGB channels
    redChannel = zeros(height, width);
    greenChannel = zeros(height, width);
    blueChannel = zeros(height, width);

    % Extract RGGB pattern components
    % Red channel (R in RGGB is at [1,1])
    redChannel(1:2:end, 1:2:end) = inImage(1:2:end, 1:2:end);
    % blueChannel(1:2:end, 1:2:end) = inImage(1:2:end, 1:2:end);
    
    % Blue channel (B in RGGB is at [2,2])
    blueChannel(2:2:end, 2:2:end) = inImage(2:2:end, 2:2:end);
    % redChannel(2:2:end, 2:2:end) = inImage(2:2:end, 2:2:end);
    
    % Green channel (G in RGGB is at [1,2] and [2,1])
    greenChannel(1:2:end, 2:2:end) = inImage(1:2:end, 2:2:end);
    greenChannel(2:2:end, 1:2:end) = inImage(2:2:end, 1:2:end);

    % Perform bilinear interpolation
    % Interpolate Red channel by using convolution
    redChannel = convolve_RB(redChannel);
    % Interpolate Green channel by using convolution
    greenChannel = convolve_G(greenChannel);
    % Interpolate Blue channel by using convolution
    blueChannel = convolve_RB(blueChannel);
    
    % Put them together
    demImage = cat(3, redChannel, greenChannel, blueChannel);

end


function channel = convolveBL(channel_in)
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

function channel = convolveBLG(channel_in)
    % Create a mask of known values
    mask = channel_in > 0;
    
    % Interpolate missing values using convolution
    kernel = [0, 1, 0; 1, 1, 1; 0, 1, 0]; % Bilinear weights
    
    % Apply convolution to the image and the mask
    interpolated = conv2(channel_in, kernel, 'same');
    weight = conv2(double(mask), kernel, 'same');
    
    % Combine interpolated values with known values
    channel = interpolated ./ weight; % Normalize interpolation by weights
    % channel(mask) = channel_in(mask);   
end

function channel = convolveNN_RB(channel_in)
    % Interpolate missing values using convolution
    kernel = [1, 1, 0; 1, 1, 0; 0, 0, 0]; % Bilinear weights
    % kernel = kernel / sum(kernel(:));    % Normalize the kernel
    
    % Apply convolution to the image and the mask
    channel = conv2(channel_in, kernel, 'same');
     
end

function channel = convolveNN_G(channel_in)
    
    % Interpolate missing values using convolution
    kernel = [0, 1, 0; 0, 1, 0; 0, 0, 0]; % Bilinear weights
    % kernel = kernel / sum(kernel(:));    % Normalize the kernel
    
    % Apply convolution to the image and the mask
    channel = conv2(channel_in, kernel, 'same');
     
end

%-------------------------------------------------------------------------
%% Ex. 4: White balancing
% Gray World Balancing
% image = gw_balancing(image);
% figure;imshow(imgGWbal);

% Gray World Balancing
% image = m_balancing(image);
% figure;imshow(imgWWbal);

% Manual Balancing
image = m_balancing(image);
% figure;imshow(image);

%figure;imshow(imgDem);

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

function balancedIm = m_balancing (inImage)
    imshow(inImage);
    % Usar ginput para capturar un clic del usuario
    [x, y] = ginput(1); % Permite un solo clic (puedes cambiar el número para más clics)

    % Convertir las coordenadas a índices enteros
    col = round(x); % Columna (width)
    row = round(y); % Fila (height)
    col
    row

    % Test Img
    % col = 2365;
    % row = 2181;
    channelSum = sum(inImage(row,col,:));
    S_R = channelSum / (3*inImage(row,col,1));
    S_G = channelSum / (3*inImage(row,col,2));
    S_B = channelSum / (3*inImage(row,col,3));

    balancer = [S_R, S_G, S_B];
    balancedIm = inImage .* reshape(balancer, 1, 1, []);
    balancedIm(row,col)

end
%-------------------------------------------------------------------------
%% Ex. 5: Denoising
% imgDenoised = denoise(image, @denoiseMean, 5);
% figure;imshow(imgDenoised);
% 
% imgDenoised = denoise(image, @denoiseMean, 10);
% figure;imshow(imgDenoised);
% 
imgDenoised = denoise(image, @denoiseMean, 10, false);
% figure;imshow(imgDenoised);

% imgDenoised = denoise(image, @denoiseMedian, 5);
% figure;imshow(imgDenoised);
% 
% imgDenoised = denoise(image, @denoiseMedian, 10);
% figure;imshow(imgDenoised);
% 
% imgDenoised = denoise(image, @denoiseMedian, 50);
% figure;imshow(imgDenoised);

% image = denoise(image, @denoiseGaussian, 10, false);
% figure;imshow(imgDenoised);


% Functions Ex. 5
function denoised = denoise(inImage, denoiseFunc, kernelSize, plotFFT) 
    % Handle optional plot flag
    if nargin < 4
        plotFFT = false;
    end

    % Denoise each channel
    redChannel = denoiseFunc(inImage(:,:,1), kernelSize, plotFFT);
    greenChannel = denoiseFunc(inImage(:,:,2), kernelSize, false);
    blueChannel = denoiseFunc(inImage(:,:,3), kernelSize, false);

    % Put them together
    denoised = cat(3, redChannel, greenChannel, blueChannel);
    
end

function denoised = denoiseMean(channel_in, kSize, plotFFT)    
    % kernel = [1, 1, 1; 1, 1, 1; 1, 1, 1]; % weights
    kernel = ones(kSize);
    kernel = kernel / (kSize*kSize);    % Normalize the kernel
    if plotFFT
        % Compute FFT for mean kernel
        kernel_pad = padarray(kernel, [10,10],0,"both");
        fft_mean = fft2(kernel_pad);
        fft_mean_shifted = fftshift(fft_mean);
        magnitude_mean = abs(fft_mean_shifted);
        log_magnitude_mean = log(1 + magnitude_mean);  % Log scale for visualization
        
        % Plot results
        figure;
        
        % Mean kernel spatial domain
        subplot(2, 2, 1);
        imagesc(kernel_pad);
        title('Mean Kernel (Spatial Domain)');
        colorbar;
        axis image;
        
        % Mean kernel frequency domain
        subplot(2, 2, 2);
        imagesc(log_magnitude_mean);
        title('Mean FFT Magnitude (Log Scale)');
        colorbar;
        axis image;
    end
    
    % Apply convolution to the image and the mask
    denoised = conv2(channel_in, kernel, 'same'); 
end

function denoised = denoiseMedian(channel_in, kSize, plotFFT)
    % % Get the dimensions of the input channel
    % [rows, cols] = size(channel_in);
    % 
    % % Initialize the denoised output
    % denoised = zeros(rows, cols);
    % 
    % % Iterate over each pixel in the original image
    % for i = 1:rows
    %     for j = 1:cols
    %         % Determine the neighborhood bounds clamped to the image edges
    %         rowStart = max(1, i - kSize);
    %         rowEnd = min(rows, i + kSize);
    %         colStart = max(1, j - kSize);
    %         colEnd = min(cols, j + kSize);
    % 
    %         % Extract the valid neighborhood window
    %         window = channel_in(rowStart:rowEnd, colStart:colEnd);
    % 
    %         % Compute the median of the window and assign to output
    %         denoised(i, j) = median(window(:));
    %     end
    % end

    % Get the dimensions of the input channel
    denoised = medfilt2(channel_in, [2 * kSize + 1, 2 * kSize + 1]);
end


function denoised = denoiseGaussian(channel_in, kSize, plotFFT)
    % Mean and std
    std = 2.;

    % Create a grid with X and Y
    [X, Y] = meshgrid(-kSize/2:kSize/2, -kSize/2:kSize/2);

    % Compute Gaussian function for the kernel
    kernel = exp(-(X.^2 + Y.^2) / (2 * std^2)) / (2 * pi * std^2);
    kernel = kernel / sum(kernel(:));
    if plotFFT
        % Compute FFT for mean kernel
        fft_mean = fft2(kernel);
        fft_mean_shifted = fftshift(fft_mean);
        magnitude_mean = abs(fft_mean_shifted);
        log_magnitude_mean = log(1 + magnitude_mean);  % Log scale for visualization
        
        % Plot results
        figure;
        
        % Mean kernel spatial domain
        subplot(2, 2, 1);
        imagesc(kernel);
        title('Gaussian Kernel (Spatial Domain)');
        colorbar;
        axis image;
        
        % Mean kernel frequency domain
        subplot(2, 2, 2);
        imagesc(log_magnitude_mean);
        title('Gaussian FFT Magnitude (Log Scale)');
        colorbar;
        axis image;
    end
    
    % Apply convolution to the image and the mask
    denoised = conv2(channel_in, kernel, 'same'); 
end

% -------------------------------------------------------------------------
%% Ex. 6: Color balance
saturationBoost = 1.8;
imgHSV = rgb2hsv(image);
S = imgHSV(:,:,2);
S = saturationBoost * S;
S(S>1) = 1;
imgHSV(:,:,2)=S;
image = hsv2rgb(imgHSV);
% figure;imshow(image);

%-------------------------------------------------------------------------
% Ex. 7: Tone reproduction
% Linear scaling
percentage = 0.15;
imgGray = rgb2gray(image);
maxGray = max(imgGray, [], "all");
% imgColor = percentage*maxGray+imgColor;
image = image*(1+percentage*maxGray);
image = min(max(image, 0), 1);
% figure;imshow(image);

% Non-Linnear GC
gamma = 1.8;
imgMinorMask = (image <= 0.0031308);
imgMajorMask = (image > 0.0031308);
imgGC = zeros(size(image));
imgGC(imgMinorMask) = 12.92*image(imgMinorMask);
imgGC(imgMajorMask) = (1+0.055)*image(imgMajorMask).^(1/gamma)-0.055;
image = imgGC;
figure;imshow(image);
% %-------------------------------------------------------------------------
% % Ex. 8: Compression
% imgFinal = image;
% imwrite(imgFinal,"imgFinal.png")
% quality = 95;
% filename = sprintf('imgFinal_%d.jpeg', quality);
% imgFinal_uint8 = uint8(imgFinal * 255);
% imwrite(imgFinal_uint8, filename, 'Quality', quality);

