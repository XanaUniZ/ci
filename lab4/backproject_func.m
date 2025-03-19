function G = backProjection_reconstruction(data, resolution_voxel, resolution_capture)
    laser_origin = data.laserOrigin;
    spad_origin = data.spadOrigin;

    x_l = data.laserPositions;
    x_s = data.spadPositions;

    v_c = data.volumePosition;
    v_s = data.volumeSize;

    % Create a 3D grid of voxel indices
    [v_i, v_j, v_k] = ndgrid(1:resolution_voxel, 1:resolution_voxel, 1:resolution_voxel);

    % Calculate the center for each voxel
    center = v_c - 0.5 * [v_s; v_s; v_s] + [v_i(:)'; v_j(:)'; v_k(:)'] * v_s / resolution_voxel;

    % Initialize G
    G = zeros(resolution_voxel, resolution_voxel, resolution_voxel);

    % Loop over l and s
    for l_i = 1:size(x_l, 1)
        for l_j = 1:size(x_l, 2)
            l = x_l(l_i, l_j, :);
            l = l(:);
            % Distance from the laser to the rellay wall
            d1 = norm(l - laser_origin(:));
            for s_i = 1:resolution_capture:size(x_s, 1)
                for s_j = 1:resolution_capture:size(x_s, 2)
                    s = x_s(s_i, s_j, :);
                    s = s(:);
                    % Distance from the relay wall to the SPAD
                    d4 = norm(spad_origin(:) - s);
                    % Distance from the voxel to the relay wall laser
                    d2 = vecnorm(bsxfun(@minus, center, l), 2, 1);
                    % Distance from the voxel to the relay wall SPAD
                    d3 = vecnorm(bsxfun(@minus, center, s), 2, 1);
                    t = (d1 + d2 + d3 + d4);
                    index = round(t / data.deltaT) + data.t0;
                    G = G + reshape(data.data(l_i, l_j, s_i, s_j, index), size(G));
                end
            end
        end
    end

    
    
end