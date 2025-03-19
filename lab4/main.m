data = load("data/Z_d=0.5_l=[1x1]_s=[256x256].mat");
% data = load("data/bunnybox_d=0.5_l=[16x16]_s=[16x16].mat");

g = backproject(data);
f_lap = fspecial3('lap');
G_lap = imfilter(g, -f_lap, 'symmetric');
% g = backProjection_reconstruction(data,resolution_voxel, resolution_capture);

vs_h = volshow(g, RenderingStyle="MaximumIntensityProjection", Colormap=hot);
vs_h.Parent.BackgroundColor = [0 0 0];
vs_h.Parent.GradientColor = [0 0 0];

function g = backproject(data)
    n_voxels = 8;
    origin = data.data.volumePosition;
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

    for i=1:n_voxels
        for j=1:n_voxels
            for k=1:n_voxels
                % Calculate coordinates (minimum corner of the voxel)
                x = origin(1) + (i-1) * delta_voxel;
                y = origin(2) + (j-1) * delta_voxel;
                z = origin(3) + (k-1) * delta_voxel;
                
                % Store coordinates in the vector
                voxel_coord = [x; y; z];
                
                for xli=1:size(xls, 1)
                    for xlj=1:size(xls, 1)
                        for xsi=1:size(xss, 1)
                            for xsj=1:size(xss, 1)
                                xl = squeeze(xls(xli, xlj, :));
                                xs = squeeze(xss(xsi, xsj, :));

                                d1 = norm(xl-lo);
                                d2 = norm(voxel_coord-xl);
                                d3 = norm(xs-voxel_coord);
                                d4 = norm(so-xs);
                                d = d1+d2+d3+d4 - data.data.t0;
                                g(i, j, k) = g(i, j, k) + H(xli,xlj,xsi,xsj,round(d/data.data.deltaT));
                            end
                        end
                    end
                end
            end
        end
    end
end
