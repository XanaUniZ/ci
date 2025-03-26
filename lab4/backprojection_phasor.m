function [g] = backprojection_phasor(data, n_voxels)
    origin = data.data.volumePosition + [0; -0.3; 0];
    volSize = 1; % data.data.volumeSize + data.data.volumeSize*0.1
    delta_voxel = volSize / n_voxels;
    g = zeros(n_voxels, n_voxels, n_voxels);
    % Calculate the minimum corner of the entire volume (starting point)
    origin = origin - volSize/2; % [x0-0.5, y0-0.5, z0-0.5]
    
    xls = data.data.laserPositions;
    xss = data.data.spadPositions;

    lo = data.data.laserOrigin;
    so = data.data.spadOrigin;

    H = data.data.data;  

    % Delta defined by spacing
    deltaT = norm(squeeze(data.data.spadPositions(2,1,:)) - squeeze(data.data.spadPositions(1,1,:)));
    lambda = 2*deltaT; 
    Omega_c = 1/lambda;
    sigma = lambda;
    H = morlet_filter_1D(H, data.data.deltaT, Omega_c, sigma);

    % PRECOMPUTE THE VOLUME
    % Generate coordinate vectors for each axis
    x_coords = origin(1) + (0:n_voxels-1) * delta_voxel;
    y_coords = origin(2) + (0:n_voxels-1) * delta_voxel;
    z_coords = origin(3) + (0:n_voxels-1) * delta_voxel;
    
    % Create 3D grids for each coordinate
    [X, Y, Z] = ndgrid(x_coords, y_coords, z_coords);
    
    % Concatenate into a 4D array (i,j,k,1:3 for x,y,z)
    volume = cat(1, ...
        permute(X, [4,1,2,3]), ...  % X becomes [1×8×8×8] → concatenate along dim 1
        permute(Y, [4,1,2,3]), ...
        permute(Z, [4,1,2,3]) ...
    );
    volume = reshape(volume, 3, []);
    

    % Loop over l and s
    for li = 1:size(xls, 1)
        for lj = 1:size(xls, 2)
            l = xls(li, lj, :);
            l = l(:);
            % Distance from the laser to the rellay wall
            d1 = norm(l - lo(:));
            for si = 1:size(xss, 1)
                for sj = 1:size(xss, 2)
                    s = xss(si, sj, :);
                    s = s(:);
                    % Distance from the relay wall to the SPAD
                    d4 = norm(so(:) - s);
                    % Distance from the voxel to the relay wall laser
                    d2 = vecnorm(bsxfun(@minus, volume, l), 2, 1);
                    % Distance from the voxel to the relay wall SPAD
                    d3 = vecnorm(bsxfun(@minus, volume, s), 2, 1);
                    t = (d1 + d2 + d3 + d4);
                    index = round(t / data.data.deltaT) - data.data.t0;
                    g = g + reshape(H(li, lj, si, sj, index), size(g));
                end
            end
        end
    end
    g = abs(g);
end


function H_filtered = morlet_filter_1D(H, deltaT, Omega_c, sigma)
    % Unpack size
    [L1, L2, S1, S2, T] = size(H);

    % Convert angular freq (Omega_c) to cyclic freq (f_c)
    f_c = Omega_c / (2*pi);

    % Sampling rate in Hz
    Fs = 1 / deltaT;

    % Create frequency axis for dimension T
    freq = (0 : T-1) * (Fs / T);
    % Move frequencies above Fs/2 into negative range (standard "wrapped" FFT freq)
    freq(freq > Fs/2) = freq(freq > Fs/2) - Fs;

    % 1) FFT along time dimension
    H_fft = fft(H, [], 5);

    % 2) Build Morlet kernel in frequency domains
    morletKernel = exp( -2 * pi^2 * sigma^2 * (freq - f_c).^2 );
    % Reshape to broadcast along the 5th dimension
    morletKernel = reshape(morletKernel, [1 1 1 1 T]);

    % 3) Multiply in frequency domain
    H_fft_filtered = H_fft .* morletKernel;

    % 4) Inverse FFT along time dimension to get H'
    H_filtered = ifft(H_fft_filtered, [], 5, 'symmetric');
end
