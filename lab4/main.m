data = load("data/Z_d=0.5_l=[1x1]_s=[256x256].mat");
% data = load("data/bunny_d=0.5_l=[1x1]_s=[256x256].mat");
% data = load("data/bunnybox_d=0.5_l=[16x16]_s=[16x16].mat");
% data = load("data/bunny_d=0.5_c=[256x256].mat");

n_voxels = 16;
tic;
% g = backprojection_naive(data, n_voxels);
% g = backprojection_fast(data, n_voxels);
% g = backprojection_confocal(data, n_voxels);
g = backprojection_attn(data, n_voxels);
% g = backprojection_phasor(data, n_voxels);
elapsed_time = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsed_time);

f_lap = fspecial3('lap');
G_lap = imfilter(g, -f_lap, 'symmetric');

vs_h = volshow(abs(G_lap), RenderingStyle="MaximumIntensityProjection", Colormap=hot);
% vs_h = volshow(sqrt(g), RenderingStyle="MaximumIntensityProjection", Colormap=hot);
vs_h.Parent.BackgroundColor = [0 0 0];
vs_h.Parent.GradientColor = [0 0 0];

