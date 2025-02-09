%% Read data
aperture = imread('apertures/circular.bmp');
image    = imread('images/penguins.jpg');
% Do NOT extract a single channel so that the image stays in color

% Noise level (gaussian noise)
sigma = 0.005;

% Blur size
blurSize = 7;

f0 = im2double(image);  % f0 is now a color image (height x width x channels)
[height, width, channels] = size(f0);

% Prior matrix: 1/f law
A_star = eMakePrior(height, width) + 1e-8;
C = sigma.^2 * height * width ./ A_star;

% Normalization (using fspecial to compute a disk filter)
temp = fspecial('disk', blurSize);
flow = max(temp(:));

% Calculate effective PSF
k1 = im2double( imresize(aperture, [2*blurSize + 1, 2*blurSize + 1], 'nearest') );
k1 = k1 * (flow / max(k1(:)));
% (Optionally, you could normalize k1 so that sum(k1(:))==1)

%% Apply blur to each channel independently
f1 = zeros(size(f0));
for c = 1:channels
    f1(:,:,c) = zDefocused(f0(:,:,c), k1, sigma, 0);
end

%% Padding aperture (this part remains unchanged)
k1P = zPSFPad(k1, max(height, width), max(height, width));

%% Aperture power spectra (only one PSF exists)
F_ap = fft2(k1P);
F_ap = fftshift(F_ap .* conj(F_ap));
S_ap = log(F_ap + eps);  % add eps to avoid log(0)

% For display, extract a central column and row
midP = round(size(S_ap,2)/2);
S_ap_X = S_ap(:, midP);
midP_row = round(size(S_ap,1)/2);
S_ap_Y = S_ap(midP_row, :);

%% Display Aperture and its Frequency Spectrum
figure;
subplot_tight(2, 2, 1, 0.05, false)
imagesc(k1P);
axis image off
title('Aperture');

subplot_tight(2, 2, 2, 0.05, false)
imagesc(S_ap);
axis image off
title('Aperture Frequency');

subplot_tight(2, 2, 3, 0.05, false)
plot(linspace(-1, 1, length(S_ap_X)), S_ap_X)
grid on
title('Normalized frequency X');

subplot_tight(2, 2, 4, 0.05, false)
plot(linspace(-1, 1, length(S_ap_Y)), S_ap_Y)
grid on
title('Normalized frequency Y');

%% Image Power Spectra (compute each channel independently)
% Preallocate arrays for the power spectra of the original and defocused images.
F0 = zeros(height, width, channels);
F1 = zeros(height, width, channels);
S0 = zeros(height, width, channels);
S1 = zeros(height, width, channels);
for c = 1:channels
    % Compute 2D FFT for each channel, take magnitude squared, and shift
    F0(:,:,c) = fftshift( fft2(f0(:,:,c)) .* conj(fft2(f0(:,:,c))) );
    F1(:,:,c) = fftshift( fft2(f1(:,:,c)) .* conj(fft2(f1(:,:,c))) );
    % Take logarithm for display (add eps to avoid log(0))
    S0(:,:,c) = log(F0(:,:,c) + eps);
    S1(:,:,c) = log(F1(:,:,c) + eps);
end

% Choose the central column/row for each channel for plotting.
midX = round(width/2);
midY = round(height/2);
x_axis = linspace(-1, 1, height);
y_axis = linspace(-1, 1, width);

%% Display Image Frequency Spectra (Overlay the channels)
% Here we overlay the frequency spectra for each color channel using red, green, and blue.
figure;

% Original image frequency in X-direction
subplot_tight(2, 2, 1, [0.1 0.05], false)
hold on;
colors = {'r','g','b'};
for c = 1:channels
    % S0(:,midX,c) is a vector (length = height)
    plot(x_axis, S0(:, midX, c), colors{c});
end
hold off;
grid on;
title('Original X');

% Original image frequency in Y-direction
subplot_tight(2, 2, 2, [0.1 0.05], false)
hold on;
for c = 1:channels
    % Squeeze to get a row vector (length = width)
    plot(y_axis, squeeze(S0(midY, :, c)), colors{c});
end
hold off;
grid on;
title('Original Y');

% Defocused image frequency in X-direction
subplot_tight(2, 2, 3, [0.1 0.05], false)
hold on;
for c = 1:channels
    plot(x_axis, S1(:, midX, c), colors{c});
end
hold off;
grid on;
title('Defocused X');

% Defocused image frequency in Y-direction
subplot_tight(2, 2, 4, [0.1 0.05], false)
hold on;
for c = 1:channels
    plot(y_axis, squeeze(S1(midY, :, c)), colors{c});
end
hold off;
grid on;
title('Defocused Y');

%% Function: zPSFPad
function outK = zPSFPad(inK, height, width)
    % Zero-pad the PSF in inK to the desired dimensions (height x width)
    [sheight, swidth] = size(inK);
    outK = zeros(height, width);
    rowStart = floor((height - sheight)/2) + 1;
    colStart = floor((width - swidth)/2) + 1;
    outK(rowStart:rowStart+sheight-1, colStart:colStart+swidth-1) = inK;
end
