function [g] = backprojection_attn(data, n_voxels)
    eps = 0.005;
    origin = data.data.volumePosition + [0; -0.3; 0];
    volSize =  data.data.volumeSize + data.data.volumeSize*0.1;
    delta_voxel = volSize / n_voxels;
    g = zeros(n_voxels, n_voxels, n_voxels);
    % Calculate the minimum corner of the entire volume (starting point)
    origin = origin - volSize/2; % [x0-0.5, y0-0.5, z0-0.5]
    
    xls = data.data.laserPositions;
    xss = data.data.spadPositions;

    lo = data.data.laserOrigin;
    so = data.data.spadOrigin;

    H = data.data.data;
    data = data.data;

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

                    % Calculate quadratic attenuation factor
                    % quadratic_attenuation = (d2 + d3) .* (d2 + d3);
                    quadratic_attenuation = (d2.^2).*(d3.^2);
                    
                    % Calculate laser-related cosine term components
                    laser_normal = squeeze(data.laserNormals(li, lj, :));
                    norm_laser_denominator = norm(laser_normal);
                    replicated_laser_normal = repmat(laser_normal, 1, size(volume, 2));
                    volume_relative_to_laser = volume - l;
                    laser_dot_product = dot(volume_relative_to_laser, replicated_laser_normal, 1);
                    cos_laser = abs(laser_dot_product ./ (d2 * norm_laser_denominator + eps));
                    
                    % Calculate SPAD-related cosine term components
                    spad_normal = squeeze(data.spadNormals(si, sj, :));
                    norm_spad_denominator = norm(spad_normal);
                    replicated_spad_normal = repmat(spad_normal, 1, size(volume, 2));
                    volume_relative_to_spad = volume - s;
                    spad_dot_product = dot(volume_relative_to_spad, replicated_spad_normal, 1);
                    cos_spad = abs(spad_dot_product ./ (d3 * norm_spad_denominator + eps));
                    
                    % Combine to get final cosine attenuation term
                    cosine_term = cos_laser .* cos_spad;

                    % Combine attenuation factors
                    attenuation = quadratic_attenuation ./ cosine_term;
                    % attenuation = quadratic_attenuation;

                    index = round(t / data.deltaT) - data.t0;
                    g = g + (reshape(H(li, lj, si, sj, index), size(g)) .* reshape(attenuation, size(g)));
                end
            end
        end
    end

end