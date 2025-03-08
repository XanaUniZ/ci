%load HDR
hdr = hdrread('../Part1/Results/hdr_image.hdr');

fignum = 6;
naive = false;

if naive
    n = "_naive";
else
    n = "";
end

for dR = [2, 4, 6, 8]

    imwrite( ...
        durand_tonemapping(hdr, dR, fignum, naive), ...
        "Results/tonemapped_durand_dR_" + dR + n + ".png" ...
        )
    
    fignum = fignum + 1;

end

clear fignum dR;
