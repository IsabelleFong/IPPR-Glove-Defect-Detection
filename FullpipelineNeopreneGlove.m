        
function [masks, counts] = FullpipelineNeopreneGlove(img, ax)

    [imgH, imgW, ~] = size(img);

    % --- Segment once at 6000px ---
    segH   = 6000;
    segW   = round(imgW * segH / imgH);
    imgSeg = imresize(img, [segH segW]);

    [glove_mask, smoothed_boundary, dist_transform, gabor_norm] = segmentGlove(imgSeg, false);

    masks = struct('burn', [], 'Cut', [], 'abrasion', []);


    % --- Abrasion at 6000px (reuse segmentation) ---
    [masks.abrasion, ~] = detectAbrasionFromPrecomputedGabor(imgSeg, glove_mask, gabor_norm, false);

    % --- Burns at 4000px ---
    burnScale = 4000 / imgH;
    imgBurn   = imresize(img, burnScale);
    
    [gm_b, sb_b, dt_b, ~] = segmentGlove(imgBurn, false);
    
    out        = detectGloveBurnsFromSeg(imgBurn, gm_b, sb_b, dt_b);
    masks.burn = out.BWkeep;
    % --- Cuts at 2000px ---
    CutScale = 2000 / imgH;
    imgCut   = imresize(img, CutScale);
    
    [gm_t, ~, dt_t, ~] = segmentGlove(imgCut, false);
    
    [masks.Cut, ~, ~] = detectCut(imgCut, gm_t, dt_t, struct(), false);
     


    counts.burn     = countDefects(masks.burn);
    counts.Cut     = countDefects(masks.Cut);
    counts.abrasion = countDefects(masks.abrasion);

        % Draw at the end, inside the same file
    drawDefectOverlay(img, masks, ax);
end







%% ====================== Glove Segmentation Function ======================
% Segments the glove from background using texture-based signals only.
% Does NOT use intensity or colour contrast — background agnostic.
% Does NOT assume fixed resolution or background colour.
%
% INPUTS:
%   img   — original RGB image
%   debug — true/false, shows intermediate results if true
%
% OUTPUTS:
%   glove_mask        — logical mask of glove region (raw, for detection)
%   smoothed_boundary — logical perimeter of smoothed mask (for geometry)
%   dist_transform    — distance of each glove pixel from smoothed boundary

function [glove_mask, smoothed_boundary, dist_transform, gabor_norm] = segmentGlove(img, debug)

    if nargin < 2
        debug = false;
    end

    glove_mask        = [];
    smoothed_boundary = [];
    dist_transform    = [];

    % gray = im2gray(img);
    % gray = im2double(gray);

    lab  = rgb2lab(img);
    gray = rescale(lab(:,:,1));  % L channel, rescaled to 0-1

    if debug
        figure('Name', 'Segmentation Debug', 'NumberTitle', 'off');
        subplot(2, 3, 1);
        imshow(img);
        title('1. Original image');
    end

    % ------------------------------------------------------------------ %
    %  SIGNAL 1 — Gabor texture response                                  %
    %  Detects periodic woven fabric pattern.                             %
    %  Wavelength is estimated adaptively from the image spectrum —       %
    %  no manual tuning needed across different weave densities.          %
    %  Fires on woven fabric zones, weak on smooth neoprene.              %
    % ------------------------------------------------------------------ %
    [gabor_mask, gabor_norm, wavelength] = getGaborMask(gray);

    if debug
        subplot(2, 3, 2);
        imagesc(gabor_norm);
        colormap(gca, 'hot');
        colorbar;
        title(sprintf('2. Gabor response (wavelength=%.1f px)', wavelength));
        axis image off;

        subplot(2, 3, 3);
        imshow(gabor_mask);
        title('3. Gabor mask');
    end

    % ------------------------------------------------------------------ %
    %  SIGNAL 2 — Local texture energy via stdfilt                        %
    %  Measures how much local variation exists in each neighbourhood.    %
    %  Glove surface — both woven and smooth neoprene — has higher        %
    %  local variation than a flat featureless background.                %
    %  Complements Gabor by catching smooth material zones that have      %
    %  no periodic structure for Gabor to lock onto.                      %
    % ------------------------------------------------------------------ %
    [texture_mask, std_norm] = getTextureMask(gray);

    if debug
        subplot(2, 3, 4);
        imagesc(std_norm);
        colormap(gca, 'hot');
        colorbar;
        title('4. Local texture energy map');
        axis image off;

        subplot(2, 3, 5);
        imshow(texture_mask);
        title('5. Texture energy mask');
    end

    % ------------------------------------------------------------------ %
    %  SIGNAL RELIABILITY CHECK                                            %
    %  A signal covering less than 5% fired on nothing useful.           %
    %  A signal covering more than 80% fired on everything including      %
    %  background — not useful for isolation.                             %
    %  Only reliable signals enter the combination.                       %
    % ------------------------------------------------------------------ %
    gabor_coverage   = sum(gabor_mask(:))   / numel(gray);
    texture_coverage = sum(texture_mask(:)) / numel(gray);

    gabor_reliable   = gabor_coverage   > 0.05 && gabor_coverage   < 0.80;
    texture_reliable = texture_coverage > 0.05 && texture_coverage < 0.80;

    if debug
        fprintf('\n--- Signal reliability ---\n');
        fprintf('Gabor:   %.1f%% coverage — %s\n', gabor_coverage*100,   reliabilityStr(gabor_reliable));
        fprintf('Texture: %.1f%% coverage — %s\n', texture_coverage*100, reliabilityStr(texture_reliable));
    end

    % ------------------------------------------------------------------ %
    %  COMBINE RELIABLE SIGNALS                                            %
    %  Union — glove detected if EITHER texture signal fires.             %
    %  This handles mixed-material surfaces (woven + smooth neoprene).   %
    % ------------------------------------------------------------------ %
    combined = false(size(gray));

    if gabor_reliable
        combined = combined | gabor_mask;
    end
    if texture_reliable
        combined = combined | texture_mask;
    end

    % Fallback — neither signal reliable, use whichever is closest to 30%
    if ~gabor_reliable && ~texture_reliable
        fprintf('Warning: neither texture signal reliable. Using best available.\n');
        coverages = [gabor_coverage, texture_coverage];
        [~, best] = min(abs(coverages - 0.30));
        masks     = {gabor_mask, texture_mask};
        combined  = masks{best};
    end

    % ------------------------------------------------------------------ %
    %  MORPHOLOGICAL CLEANUP                                               %
    %  All size parameters expressed as image fractions —                 %
    %  resolution independent.                                            %
    % ------------------------------------------------------------------ %

    % Remove small isolated noise regions
    min_area = 0.01 * numel(gray);
    combined = bwareaopen(combined, round(min_area));

    % Fill internal holes (finger gaps, palm opening, open back cutout)
    combined = imfill(combined, 'holes');

    % Close small gaps between nearby glove material regions
    close_radius = max(5, round(size(gray, 1) * 0.10));
    combined     = imclose(combined, strel('disk', close_radius));
    combined = imfill(combined, 'holes');

    % ------------------------------------------------------------------ %
    %  KEEP LARGEST CONNECTED COMPONENT                                    %
    %  Glove is the dominant textured object in the image.                %
    % ------------------------------------------------------------------ %
    cc = bwconncomp(combined);
    if cc.NumObjects == 0
        fprintf('Segmentation failed: no regions found.\n');
        fprintf('Check image lighting or capture against a lighter background.\n');
        return;
    end

    stats      = regionprops(cc, 'Area', 'PixelIdxList');
    [~, idx]   = max([stats.Area]);
    glove_mask = false(size(gray));
    glove_mask(stats(idx).PixelIdxList) = true;
    glove_mask = imfill(glove_mask, 'holes');

    % Coverage sanity check
    glove_coverage = sum(glove_mask(:)) / numel(glove_mask);
    if glove_coverage < 0.05
        fprintf('Segmentation warning: mask covers only %.1f%% of image.\n', glove_coverage*100);
        fprintf('Image may be underexposed or background texture too similar to glove.\n');
    else
        fprintf('Glove segmented: %.1f%% of image area.\n', glove_coverage*100);
    end

    % ------------------------------------------------------------------ %
    %  BOUNDARY SMOOTHING                                                  %
    %  Raw boundary follows surface damage and fraying — unreliable for   %
    %  geometric distance measurements used in defect classification.     %
    %  Smoothed boundary reflects the structural glove shape.             %
    %  Detection still runs on original unsmoothed glove_mask.            %
    % ------------------------------------------------------------------ %
    smooth_radius     = max(20, round(size(gray, 1) * 0.02));
    smoothed_mask     = imclose(glove_mask, strel('disk', smooth_radius));
    smoothed_mask     = imfill(smoothed_mask, 'holes');
    smoothed_boundary = bwperim(smoothed_mask);

    % ------------------------------------------------------------------ %
    %  DISTANCE TRANSFORM                                                  %
    %  Each glove pixel gets its distance to the nearest smoothed         %
    %  boundary pixel. Used downstream to classify whether a defect       %
    %  region is boundary-adjacent or interior.                           %
    % ------------------------------------------------------------------ %
    dist_transform = bwdist(smoothed_boundary);

    if debug
        subplot(2, 3, 6);
        overlay          = img;
        g_ch             = double(overlay(:,:,2));
        g_ch(glove_mask) = min(255, g_ch(glove_mask) + 80);
        overlay(:,:,2)   = uint8(g_ch);
        imshow(overlay);
        title(sprintf('6. Final mask overlay (%.1f%%)', glove_coverage*100));

        figure('Name', 'Boundary + Distance Transform', 'NumberTitle', 'off');
        subplot(1, 2, 1);
        imshow(smoothed_boundary);
        title('Smoothed boundary');

        subplot(1, 2, 2);
        imagesc(dist_transform .* double(glove_mask));
        colormap(gca, 'jet');
        colorbar;
        title('Distance from boundary (glove pixels only)');
        axis image off;
    end

    fprintf('Segmentation complete.\n');

end


%% ====================== Gabor Texture Mask ================================
% Applies Gabor filter bank at multiple orientations.
% Wavelength estimated adaptively from image power spectrum.
% Strong response = periodic woven fabric structure present.

function [gabor_mask, gabor_norm, wavelength] = getGaborMask(gray)

    orientations   = [0, 45, 90, 135];
    wavelength     = estimateWeaveWavelength(gray);

    gabor_response = zeros(size(gray));
    for theta = orientations
        g              = gabor(wavelength, theta);
        [mag, ~]       = imgaborfilt(double(gray), g);
        gabor_response = gabor_response + mag;
    end

    gabor_norm = rescale(gabor_response);
    T          = graythresh(gabor_norm);
    gabor_mask = gabor_norm > T;

end


%% ====================== Local Texture Energy Mask =========================
% Local standard deviation measures neighbourhood pixel variation.
% Higher variation = more surface structure = likely glove material.
% Works on smooth neoprene zones where no periodic weave exists,
% as long as the surface has more micro-variation than the background.
% Window size scales with image height for resolution independence.

function [texture_mask, std_norm] = getTextureMask(gray)

    win_size = max(7, round(size(gray, 1) * 0.01));
    if mod(win_size, 2) == 0
        win_size = win_size + 1;
    end

    local_std    = stdfilt(gray, true(win_size));
    std_norm     = rescale(local_std);
    T            = graythresh(std_norm);
    texture_mask = std_norm > T;

end


%% ====================== Weave Wavelength Estimation =======================
% Estimates dominant spatial frequency of woven fabric from image spectrum.
% Weave produces distinct peaks in 2D FFT power spectrum.
% Peak distance from centre = spatial frequency = reciprocal of wavelength.
% DC component and high-frequency noise suppressed before peak finding.
% Result clamped to sensible range for exercise gloves (4–40 pixels).

function wavelength = estimateWeaveWavelength(gray)

    F       = fft2(double(gray));
    F_shift = fftshift(F);
    power   = abs(F_shift).^2;

    [rows, cols] = size(power);
    centre_r     = round(rows / 2);
    centre_c     = round(cols / 2);
    [X, Y]       = meshgrid(1:cols, 1:rows);

    % Suppress DC component
    suppress_radius = round(min(rows, cols) * 0.05);
    dc_mask         = sqrt((X-centre_c).^2 + (Y-centre_r).^2) < suppress_radius;
    power(dc_mask)  = 0;

    % Suppress sensor noise frequencies
    max_freq_radius  = round(min(rows, cols) * 0.4);
    noise_mask       = sqrt((X-centre_c).^2 + (Y-centre_r).^2) > max_freq_radius;
    power(noise_mask) = 0;

    [~, peak_idx]    = max(power(:));
    [peak_r, peak_c] = ind2sub(size(power), peak_idx);

    freq_distance    = sqrt((peak_r-centre_r)^2 + (peak_c-centre_c)^2);
    cycles_per_pixel = freq_distance / min(rows, cols);
    wavelength_est   = 1 / cycles_per_pixel;

    wavelength = max(4, min(40, wavelength_est));
    fprintf('Estimated weave wavelength: %.1f pixels\n', wavelength);

end


%% ====================== Helper ============================================
function s = reliabilityStr(flag)
    if flag
        s = 'RELIABLE';
    else
        s = 'unreliable';
    end
end



function [Cut_mask, Cut_stats, debugOut] = detectCut(img, glove_mask, dist_transform, opts, debug)

    if nargin < 4 || isempty(opts); opts = struct; end
    if nargin < 5;                  debug = false;  end

    % ------------------------------------------------------------------ %
    %  PARAMETERS                                                          %
    %  All tunable thresholds in one place.                               %
    % ------------------------------------------------------------------ %

    % Skin detection in HSV
    % Hue:        0.0 – 0.12  (red/orange/yellow skin tones)
    % Saturation: 0.15 – 0.65 (colourful but not vivid)
    %             0.15 minimum excludes near-neutral backgrounds (grey/white
    %             floor tiles, white paper) which have very low saturation
    % Value:      0.30 – 1.00 (not too dark)
    opts = setDefault(opts, 'skinHueMin',   0.00);
    opts = setDefault(opts, 'skinHueMax',   0.12);
    opts = setDefault(opts, 'skinSatMin',   0.23);   % raised from 0.10 — rejects grey backgrounds
    opts = setDefault(opts, 'skinSatMax',   0.65);
    opts = setDefault(opts, 'skinValMin',   0.30);
    opts = setDefault(opts, 'maxExtent', 0.60);

    % Boundary exclusion — ignore skin within this many pixels of the
    % glove edge (fraying and wrist exposure, not interior Cuts)
   
    opts = setDefault(opts, 'minDistFromEdge', 80);

    % Size filter — absolute pixel areas.
    % minArea: ~50px  — removes tiny noise blobs
    % maxArea: ~40000px — rejects large intentional design openings
    opts = setDefault(opts, 'minArea',  2000);
    opts = setDefault(opts, 'maxArea',  40000);

    % Shape filter — slashes are elongated; round openings are not
    % Aspect ratio:  major/minor axis length ratio
    % Eccentricity:  0 = perfect circle, 1 = line — slashes score > 0.75
    % Circularity:   4*pi*A/P^2 — 1 = circle, lower = irregular/elongated
    % A blob must pass EITHER eccentricity OR aspect ratio (union),
    % AND must pass circularity (always required).
    opts = setDefault(opts, 'minAspectRatio',  1.4);
    opts = setDefault(opts, 'minEccentricity', 0.75);
    opts = setDefault(opts, 'maxCircularity',  0.55);
    opts = setDefault(opts, 'minSolidity',     0.55); 
    opts = setDefault(opts, 'maxThickness', 60);  % minor axis length in pixels at 2000px height% real Cuts are solid filled blobs
                                                        % U-shapes/curved thumb blobs score lower

    % ------------------------------------------------------------------ %
    %  STEP 1 — SKIN DETECTION IN HSV                                     %
    % ------------------------------------------------------------------ %
    hsv        = rgb2hsv(img);
    H          = hsv(:,:,1);
    S          = hsv(:,:,2);
    V          = hsv(:,:,3);

    skin_mask  = (H >= opts.skinHueMin) & (H <= opts.skinHueMax) & ...
                 (S >= opts.skinSatMin) & (S <= opts.skinSatMax) & ...
                 (V >= opts.skinValMin);

    % ------------------------------------------------------------------ %
    %  STEP 2 — RESTRICT TO INTERIOR OF GLOVE MASK                       %
    %  Skin outside the glove mask = wrist/fingers sticking out —        %
    %  not a defect.                                                      %
    %  Skin near the glove boundary = fraying exposure — not a Cut.     %
    % ------------------------------------------------------------------ %
    interior_mask = glove_mask & (dist_transform >= opts.minDistFromEdge);
    skin_interior = skin_mask & interior_mask;

    % ------------------------------------------------------------------ %
    %  STEP 3 — MORPHOLOGICAL CLEANUP                                     %
    %  Small close to merge nearby skin pixels into coherent blobs.      %
    % ------------------------------------------------------------------ %
    skin_clean = imclose(skin_interior, strel('disk', 3));
    skin_clean = imfill(skin_clean, 'holes');
    skin_clean = bwareaopen(skin_clean, opts.minArea);
    
    % ------------------------------------------------------------------ %
    %  STEP 4 — SHAPE FILTERING                                           %
    %  Keep blobs that look like Cuts (elongated, irregular).           %
    %  Reject blobs that look like intentional design openings (round,   %
    %  large, regular, thick).                                                   %
    % ------------------------------------------------------------------ %
    max_area   = opts.maxArea;
    CC         = bwconncomp(skin_clean);
    R          = regionprops(CC, 'Area', 'MajorAxisLength', 'Extent','MinorAxisLength', ...
                                 'Perimeter', 'Eccentricity', 'Solidity', 'PixelIdxList');

    Cut_mask  = false(size(glove_mask));

    for k = 1 : CC.NumObjects
        area  = R(k).Area;
        maj   = R(k).MajorAxisLength;
        mn    = max(R(k).MinorAxisLength, 1);
        perim = max(R(k).Perimeter,       1);
        ecc   = R(k).Eccentricity;
        sol   = R(k).Solidity;
        ext = R(k).Extent;

        aspect      = maj / mn;
        circularity = (4 * pi * area) / (perim^2);

        is_right_size = area >= opts.minArea && area <= max_area;
        is_elongated  = aspect >= opts.minAspectRatio || ecc >= opts.minEccentricity;
        is_irregular  = circularity <= opts.maxCircularity;
        is_solid      = sol >= opts.minSolidity;
        is_not_boxy = ext <= opts.maxExtent;
        is_thin = mn <= opts.maxThickness;

        if is_right_size && is_elongated && is_irregular && is_solid && is_not_boxy && is_thin
            Cut_mask(R(k).PixelIdxList) = true;
        end
    end
    
    Cut_mask = imclose(Cut_mask, strel('disk', 15));
    Cut_mask = imfill(Cut_mask, 'holes');

    % Final regionprops on confirmed Cuts
    Cut_stats = regionprops(Cut_mask, 'Area', 'BoundingBox', 'Centroid', ...
                                        'MajorAxisLength', 'MinorAxisLength', ...
                                        'Eccentricity');

    % ------------------------------------------------------------------ %
    %  DEBUG OUTPUTS                                                       %
    % ------------------------------------------------------------------ %
    debugOut = struct;
    debugOut.skin_mask     = skin_mask;
    debugOut.interior_mask = interior_mask;
    debugOut.skin_interior = skin_interior;
    debugOut.skin_clean    = skin_clean;
    debugOut.Cut_mask     = Cut_mask;

    if debug
        figure('Name', 'Cut Detection Debug', 'NumberTitle', 'off');

        subplot(2, 3, 1);
        imshow(img);
        title('1. Original');

        subplot(2, 3, 2);
        imshow(skin_mask);
        title('2. Raw skin mask (full image)');

        subplot(2, 3, 3);
        imshow(skin_interior);
        title(sprintf('3. Skin inside glove interior (dist >= %d)', opts.minDistFromEdge));

        subplot(2, 3, 4);
        imshow(skin_clean);
        title('4. After morphological cleanup');

        subplot(2, 3, 5);
        imshow(labeloverlay(img, Cut_mask, 'Transparency', 0.45, 'Colormap', [1 0.2 0.2]));
        title('5. Confirmed Cuts (red overlay)');

    %     subplot(2, 3, 6);
    %     imshow(img);
    %     hold on;
    %     for k = 1 : numel(Cut_stats)
    %         rectangle('Position', Cut_stats(k).BoundingBox, ...
    %                   'EdgeColor', [1 0.2 0.2], 'LineWidth', 2);
    %         c = Cut_stats(k).Centroid;
    %         text(c(1), c(2), sprintf('T%d', k), ...
    %             'Color', 'yellow', 'FontSize', 11, 'FontWeight', 'bold', ...
    %             'HorizontalAlignment', 'center');
    %     end
    %     hold off;
    %     title(sprintf('6. Detections (%d found)', numel(Cut_stats)));
    end

    fprintf('Cut detection complete: %d region(s) found.\n', numel(Cut_stats));

end


%% ====================== Helper ============================================
function s = setDefault(s, field, val)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = val;
    end
end

%% ====================== Burn Defects ============================================

function out = detectGloveBurnsFromSeg(I, glove_mask, smoothed_boundary, dist_transform, opts)
% detectGloveBurnsFromSeg
% Burn / scorch detection using an externally supplied glove segmentation.
%
% Inputs:
%   I                 - RGB image or filename
%   glove_mask        - logical mask from segmentGlove
%   smoothed_boundary - logical boundary from segmentGlove
%   dist_transform    - distance transform from segmentGlove
%   opts              - optional settings struct
%
% Example:
%   [glove_mask, smoothed_boundary, dist_transform] = segmentGlove(I, true);
%   out = detectGloveBurnsFromSeg(I, glove_mask, smoothed_boundary, dist_transform);
if ischar(I) || isstring(I)
    I = imread(I);
end
I = im2double(I);
if nargin < 5
    opts = struct;
end
opts = fillDefaultOpts(opts, size(I));
% ---------- validate masks ----------
glove_mask = logical(glove_mask);
smoothed_boundary = logical(smoothed_boundary);
if isempty(glove_mask) || ~any(glove_mask(:))
    error('glove_mask is empty or invalid.');
end
if isempty(smoothed_boundary) || ~any(smoothed_boundary(:))
    error('smoothed_boundary is empty or invalid.');
end
if isempty(dist_transform)
    error('dist_transform is empty.');
end
% ---------- geometry from your segmentation ----------
borderBand = glove_mask & (dist_transform <= opts.borderDist);
safeMask   = glove_mask & (dist_transform >= opts.safeDist);
% optional: if safe region is too small, relax it
safeFrac = nnz(safeMask) / max(nnz(glove_mask),1);
if safeFrac < 0.15
    safeMask = glove_mask & (dist_transform >= max(3, round(opts.safeDist * 0.5)));
end
% ---------- color spaces ----------
labI = rgb2lab(I);
L = mat2gray(labI(:,:,1));
a = labI(:,:,2);
b = labI(:,:,3);
grayI = rgb2gray(I);
hsvI  = rgb2hsv(I);
S = hsvI(:,:,2);
V = hsvI(:,:,3);

% ---------- robust trim/logo suppression (color independent) ----------
% brightness anomaly
brightMask = glove_mask & V > prctile(V(glove_mask), opts.brightPrctile);
% texture smoothness (logos and trim are usually smooth)
localStd = stdfilt(grayI, true(7));
smoothMask = localStd < 0.02;
% intensity uniformity (logos tend to be very uniform)
Lvar = stdfilt(L, true(7));
uniformMask = Lvar < 0.015;
% border proximity (trim often sits near glove edges)
edgeMask = borderBand & smoothMask;
% combine structural cues instead of color
trimLogoMask = glove_mask & ...
    ((brightMask & smoothMask) | ...
     (uniformMask & smoothMask) | ...
     edgeMask);
% clean mask
trimLogoMask = imopen(trimLogoMask, strel('disk', opts.trimOpenRadius));
expanded = imdilate(trimLogoMask, strel('disk',1));
trimLogoMask = expanded & smoothMask;
analysisMask = glove_mask & ~trimLogoMask;
% ---------- local anomaly cues ----------
Lmed = medfilt2(L, [opts.localWin opts.localWin], 'symmetric');
Lres = abs(L - Lmed);
localStd = stdfilt(grayI, true(opts.texWin));
localStd = mat2gray(localStd);
Tmed = medfilt2(localStd, [opts.texBaseWin opts.texBaseWin], 'symmetric');
Tres = abs(localStd - Tmed);
darkCue  = mat2gray(Lmed - L);
brownCue = mat2gray(rescalePositive(b - a));
glossCue = mat2gray(imtophat(L, strel('disk', opts.topHatRadius)));
% mask everything outside glove
Lres(~glove_mask) = 0;
Tres(~glove_mask) = 0;
darkCue(~glove_mask) = 0;
brownCue(~glove_mask) = 0;
glossCue(~glove_mask) = 0;
% ---------- robust normalization using safe interior ----------
Lz = max(robustZ(Lres, safeMask), 0);
Tz = max(robustZ(Tres, safeMask), 0);
Dz = max(robustZ(darkCue, safeMask), 0);
Bz = max(robustZ(brownCue, safeMask), 0);
Gz = max(robustZ(glossCue, safeMask), 0);
% ---------- distance weighting ----------
% downweight border responses rather than hard-removing everything
depthWeight = min(dist_transform / opts.safeDist, 1);
depthWeight(~glove_mask) = 0;
score = ...
    opts.wL * Lz + ...
    opts.wT * Tz + ...
    opts.wDark * Dz + ...
    opts.wBrown * Bz + ...
    opts.wGloss * Gz;
score = score .* depthWeight;
score(~analysisMask) = 0;
score = imgaussfilt(score, opts.scoreSigma);
% ---------- threshold ----------
if opts.useAdaptiveThreshold
    T = adaptthresh(mat2gray(score), opts.adaptSensitivity);
    BWraw = imbinarize(mat2gray(score), T);
else
    BWraw = score > opts.fixedScoreThresh;
end

BWraw = bwareaopen(BWraw, opts.minArea);
% BWraw = BWraw & ~borderBand;

% Step 2: strip border pixels BEFORE closing can merge them inward
% stripRadius = 100;
% innerMask = imerode(glove_mask, strel('disk', stripRadius));
% BWraw = BWraw & innerMask;

% ---------- delete ----------
% figure('Name', 'Debug: BWraw before border filter', 'Color', 'w');
% tiledlayout(1,2, 'Padding','compact');
% nexttile; imshow(BWraw); title('BWraw after cleanup');
% nexttile; imshowpair(BWraw, borderBand, 'blend'); title('BWraw overlaid on borderBand');





BWraw = imclose(BWraw, strel('disk', 8));  
BWraw = imfill(BWraw, 'holes');
BWraw = imopen(BWraw, strel('disk', 2));
BWraw = bwareaopen(BWraw, opts.minArea * 2);

% figure('Name', 'Debug: BWraw final before regionprops', 'Color', 'w');
% tiledlayout(1,2, 'Padding','compact');
% nexttile; imshow(BWraw); title('BWraw into regionprops');
% nexttile; imshowpair(BWraw, score, 'blend'); title('BWraw overlaid on score map');

CC = bwconncomp(BWraw);
R = regionprops(CC, score, ...
    'Area','Eccentricity','Solidity','Extent', ...
    'MajorAxisLength','MinorAxisLength', ...
    'MeanIntensity','MaxIntensity','BoundingBox','Centroid','PixelIdxList');
% ---------- adaptive thresholds from raw candidates ----------
if ~isempty(R)
    allAreas  = [R.Area];
    allScores = [R.MeanIntensity];
    gloveArea = nnz(glove_mask);
    baseMinArea = max(12, round(0.00002 * gloveArea));

    % prescreen by fixed area bounds to exclude outline/strap blobs
    % before computing adaptive thresholds
    areaValidIdx = allAreas >= baseMinArea & allAreas <= opts.maxArea;

    if any(areaValidIdx)
        threshAreas  = allAreas(areaValidIdx);
        threshScores = allScores(areaValidIdx);
    else
        % fallback if nothing passes — use everything
        threshAreas  = allAreas;
        threshScores = allScores;
    end

    adaptiveMinArea      = max(baseMinArea, round(prctile(threshAreas, 20)));
    adaptiveMinMeanScore = max(0.2, prctile(threshScores, 40));  % lowered floor from 0.45
else
    adaptiveMinArea      = opts.minArea;
    adaptiveMinMeanScore = opts.minMeanScore;
end

keep = false(numel(R),1);
for k = 1:numel(R)
    A = R(k).Area;
    E = R(k).Eccentricity;
    Sol = R(k).Solidity;
    Ext = R(k).Extent;
    MI = R(k).MeanIntensity;
    Ma = R(k).MajorAxisLength;
    Mi = max(R(k).MinorAxisLength, eps);
    aspect = Ma / Mi;
    pix = R(k).PixelIdxList;
    trimFrac = mean(trimLogoMask(pix));
    borderFrac = mean(borderBand(pix));
    meanDepth = mean(dist_transform(pix));
    areaOK   = A >= adaptiveMinArea && A <= opts.maxArea;
    shapeOK  = Sol >= opts.minSolidity && E <= opts.maxEccentricity && aspect <= opts.maxAspect;
    scoreOK  = MI >= adaptiveMinMeanScore;
    extentOK = Ext >= opts.minExtent;
    trimOK   = trimFrac <= opts.maxTrimFraction;
    borderOK = borderFrac <= opts.maxBorderFraction || meanDepth >= opts.minMeanDepth;
    keep(k) = areaOK && shapeOK && scoreOK && extentOK && trimOK && borderOK;
end

% Visual debug — colour each blob by which filter killed it
debugImg = zeros([size(BWraw,1), size(BWraw,2), 3]);

for k = 1:numel(R)
    pix = R(k).PixelIdxList;
    
    A = R(k).Area;
    E = R(k).Eccentricity;
    Sol = R(k).Solidity;
    Ext = R(k).Extent;
    MI = R(k).MeanIntensity;
    Ma = R(k).MajorAxisLength;
    Mi = max(R(k).MinorAxisLength, eps);
    aspect = Ma / Mi;
    trimFrac = mean(trimLogoMask(pix));
    borderFrac = mean(borderBand(pix));
    meanDepth = mean(dist_transform(pix));

    areaOK   = A >= adaptiveMinArea && A <= opts.maxArea;
    shapeOK  = Sol >= opts.minSolidity && E <= opts.maxEccentricity && aspect <= opts.maxAspect;
    scoreOK  = MI >= adaptiveMinMeanScore;
    extentOK = Ext >= opts.minExtent;
    trimOK   = trimFrac <= opts.maxTrimFraction;
    borderOK = borderFrac <= opts.maxBorderFraction || meanDepth >= opts.minMeanDepth;

    if keep(k)
        col = [0 1 0];       % green — survived
    elseif ~areaOK
        col = [1 0 0];       % red — area
    elseif ~shapeOK
        col = [1 0.5 0];     % orange — shape
    elseif ~scoreOK
        col = [1 1 0];       % yellow — score
    elseif ~extentOK
        col = [0 0 1];       % blue — extent
    elseif ~trimOK
        col = [1 0 1];       % magenta — trim
    elseif ~borderOK
        col = [0 1 1];       % cyan — border
    else
        col = [1 1 1];       % white — unknown
    end

    debugImg(pix) = col(1);
    debugImg(pix + size(BWraw,1)*size(BWraw,2)) = col(2);
    debugImg(pix + 2*size(BWraw,1)*size(BWraw,2)) = col(3);
end

% figure('Name', 'Debug: filter cause per blob', 'Color', 'w');
% tiledlayout(1,2, 'Padding','compact');
% nexttile; imshow(debugImg);
% title('Blobs coloured by filter cause');
% nexttile; imshow(I);
% title('Original image for reference');
% 
% figure('Name', 'Debug: filter cause per blob', 'Color', 'w');
% tiledlayout(1,2, 'Padding','compact');
% nexttile; imshow(debugImg); 
% title('Blobs coloured by filter cause');
% nexttile; imshow(I); 
% title('Original image for reference');

% legend
% annotation('textbox',[0.01 0.01 0.15 0.18],'String', ...
%     {'Green = kept', 'Red = area', 'Orange = shape', ...
%      'Yellow = score', 'Blue = extent', ...
%      'Magenta = trim', 'Cyan = border'}, ...
%     'FitBoxToText','on','BackgroundColor','w');

BWkeep = false(size(BWraw));
for k = 1:numel(R)
    if keep(k)
        BWkeep(R(k).PixelIdxList) = true;
    end
end
% figure('Name', 'Debug: score and analysisMask', 'Color', 'w');
% tiledlayout(1,3, 'Padding','compact');
% nexttile; imshow(analysisMask); title('analysisMask');
% nexttile; imshow(mat2gray(score)); title('Score map');
% nexttile; 
% scoreMasked = score .* double(analysisMask);
% imshow(mat2gray(scoreMasked)); 
% colormap(gca, 'hot'); 
% title('Score within analysisMask');
% ---------- Further operations after filtering ----------
BWkeep = imclose(BWkeep, strel('disk', 15));
BWkeep = imfill(BWkeep, 'holes');
BWkeep = bwareaopen(BWkeep, 4000);

props = regionprops(BWkeep, score, ...
    'Area','Eccentricity','Solidity','Extent', ...
    'MeanIntensity','MaxIntensity','BoundingBox','Centroid');
overlay = I;
if ~isempty(props)
    boxes = vertcat(props.BoundingBox);
    overlay = insertShape(im2uint8(overlay), 'Rectangle', boxes, ...
        'Color', 'yellow', 'LineWidth', 3);
    overlay = im2double(overlay);
end
if opts.showDebug
    makeDebugFiguresSeg(I, glove_mask, smoothed_boundary, dist_transform, ...
        safeMask, borderBand, trimLogoMask, ...
        Lres, Tres, darkCue, brownCue, glossCue, score, BWraw, BWkeep, overlay);
end
out = struct;
out.BWraw = BWraw;
out.BWkeep = BWkeep;
out.props = props;
out.score = score;
out.overlay = overlay;
out.maps = struct('Lres',Lres,'Tres',Tres,'darkCue',darkCue, ...
    'brownCue',brownCue,'glossCue',glossCue);
out.masks = struct('glove_mask',glove_mask,'smoothed_boundary',smoothed_boundary, ...
    'borderBand',borderBand,'safeMask',safeMask,'trimLogoMask',trimLogoMask, ...
    'analysisMask',analysisMask);
out.dist_transform = dist_transform;
end
function opts = fillDefaultOpts(opts, imSize)
d = struct;
d.borderDist          = 12;
d.safeDist            = 28;
d.brightPrctile       = 82;
d.lowSatThresh        = 0.33;
d.trimOpenRadius      = 2;
d.trimCloseRadius     = 6;
d.localWin            = 31;
d.texWin              = 9;
d.texBaseWin          = 31;
d.topHatRadius        = 9;
d.wL                  = 0.33;
d.wT                  = 0.25;
d.wDark               = 0.22;
d.wBrown              = 0.10;
d.wGloss              = 0.10;
d.scoreSigma          = 1.3;
d.useAdaptiveThreshold = false;
d.adaptSensitivity    = 0.50;
d.fixedScoreThresh    = 1.20;
imArea = imSize(1) * imSize(2);
d.minArea             = max(40, round(imArea * 0.00003));
d.maxArea             = max(2500, round(imArea * 0.010));
d.postOpenRadius      = 1;
d.postCloseRadius     = 4;
d.minSolidity         = 0.45;
d.maxEccentricity     = 0.96;
d.maxAspect           = 4.0;
d.minMeanScore        = 0.85;
d.minExtent           = 0.20;
d.maxTrimFraction     = 0.50;
d.maxBorderFraction   = 0.45;
d.minMeanDepth        = 10;
d.showDebug           = true;
fn = fieldnames(d);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i})
        opts.(fn{i}) = d.(fn{i});
    end
end
end
function z = robustZ(X, mask)
vals = X(mask);
vals = vals(isfinite(vals));
medv = median(vals);
madv = mad(vals, 1);
if madv < 1e-6
    madv = std(vals) + 1e-6;
end
z = (X - medv) ./ (1.4826 * madv + 1e-6);
end
function Y = rescalePositive(X)
X = X - min(X(:));
mx = max(X(:));
if mx > 0
    Y = X / mx;
else
    Y = X;
end
end
function makeDebugFiguresSeg(I, glove_mask, smoothed_boundary, dist_transform, ...
    safeMask, borderBand, trimLogoMask, ...
    Lres, Tres, darkCue, brownCue, glossCue, score, BWraw, BWkeep, overlay)
% figure('Name','Burn detection debug A','Color','w');
% tiledlayout(2,3, 'Padding','compact', 'TileSpacing','compact');
% nexttile; imshow(I); title('Input RGB');
% nexttile; imshow(glove_mask); title('Glove mask');
% nexttile; imshow(smoothed_boundary); title('burn deteSmoothed boundary');
% nexttile; imagesc(dist_transform .* double(glove_mask)); axis image off; colorbar; title('Distance transform');
% nexttile; imshow(safeMask); title('Safe interior');
% nexttile; imshow(borderBand); title('Border band');
% figure('Name','Burn detection debug B','Color','w');
% tiledlayout(2,3, 'Padding','compact', 'TileSpacing','compact');
% nexttile; imshow(trimLogoMask); title('Suppressed trim/logo');
% nexttile; imshow(mat2gray(Lres)); title('Local lightness residual');
% nexttile; imshow(mat2gray(Tres)); title('Texture residual');
% % nexttile; imshow(mat2gray(darkCue)); title('Dark cue');
% % nexttile; imshow(mat2gray(brownCue)); title('Brown cue');
% % nexttile; imshow(mat2gray(glossCue)); title('Gloss cue');
% figure('Name','Burn detection debug C','Color','w');
% tiledlayout(1,3, 'Padding','compact', 'TileSpacing','compact');
% nexttile; imshow(mat2gray(score)); title('Final score');
% nexttile; imshowpair(BWraw, BWkeep, 'montage'); title('Raw | Filtered');
% nexttile; imshow(overlay); title('Overlay');
end

%% ====================== Abrasion Defect ============================================

function [abrasion_mask, debug_data] = detectAbrasionFromPrecomputedGabor(img, glove_mask, gabor_norm, debug)
% Fast abrasion detector using precomputed Gabor response from segmentGlove
%
% INPUTS:
%   img        - original image, only used for debug overlay
%   glove_mask - logical mask from segmentGlove
%   gabor_norm - normalized Gabor response from segmentGlove
%   debug      - true/false
%
% OUTPUTS:
%   abrasion_mask - logical full-size abrasion mask
%   debug_data    - struct of intermediate results

    if nargin < 4
        debug = false;
    end

    glove_mask    = logical(glove_mask);
    abrasion_mask = false(size(glove_mask));
    debug_data    = struct();

    if ~any(glove_mask(:))
        warning('Empty glove mask.');
        return;
    end

    % ------------------------------------------------------------
    % 1. Crop to glove ROI
    % ------------------------------------------------------------
    % stats = regionprops(glove_mask, 'BoundingBox');
    % bbox  = stats(1).BoundingBox;
    % 
    % x1 = max(1, floor(bbox(1)));
    % y1 = max(1, floor(bbox(2)));
    % x2 = min(size(glove_mask,2), ceil(bbox(1) + bbox(3) - 1));
    % y2 = min(size(glove_mask,1), ceil(bbox(2) + bbox(4) - 1));
    % 
    % mask_roi  = glove_mask(y1:y2, x1:x2);
    % gabor_roi = gabor_norm(y1:y2, x1:x2);
    mask_roi  = glove_mask;
    gabor_roi = gabor_norm;

    % ------------------------------------------------------------
    % 2. Threshold only glove pixels inside ROI
    % ------------------------------------------------------------
    

    % if isempty(vals)
    %     warning('No glove pixels in ROI.');
    %     return;
    % end

   T = 0.30;   % example only
   abrasion_roi = (gabor_roi > T) & mask_roi;


    % ------------------------------------------------------------
    % 3. Cleanup
    % ------------------------------------------------------------
    % min_area = max(20, round(0.0003 * nnz(mask_roi)));
    % abrasion_roi = bwareaopen(abrasion_roi, min_area);
    % abrasion_roi = imclose(abrasion_roi, strel('disk', 2));

    se_small = strel('disk', 2);   % tune: 2–5 px depending on glove resolution

% Opening first — kills isolated speckle and disconnects weak bridges
abrasion_mask = imopen(abrasion_roi, se_small);

% Area filter — removes anything opening left behind that's still too small

min_area      = 6000;
fprintf('min_area: %d px\n', min_area);
abrasion_mask = bwareaopen(abrasion_mask, min_area);

% Closing last — fills small holes and smooths blob edges
abrasion_mask = imclose(abrasion_mask, strel('disk', 20));

% ------------------------------------------------------------
% 3b. Fill and filter for bounding box readiness
% ------------------------------------------------------------
% Fill holes inside each blob so bounding box encloses solid region
abrasion_mask = imfill(abrasion_mask, 'holes');

% Remove blobs too small to be meaningful defects
% 0.001 = 0.1% of glove area — tune this threshold
min_defect_area = 20000;

fprintf('min_area: %d px\n', min_defect_area);
abrasion_mask   = bwareaopen(abrasion_mask, min_defect_area);

    % min_area     = max(20, round(0.0003 * nnz(glove_mask)));
    % abrasion_mask = bwareaopen(abrasion_roi, min_area);
    % abrasion_mask = imclose(abrasion_mask, strel('disk', 2));

    % ------------------------------------------------------------
    % 4. Put back into full image
    % ------------------------------------------------------------
    % abrasion_mask(y1:y2, x1:x2) = abrasion_roi;
    stats = regionprops(abrasion_mask, 'Area', 'Solidity', 'PixelIdxList');

    for i = 1:length(stats)
        % Solidity = Area / ConvexHullArea
        % Compact blobs = close to 1.0, squiggly/thin blobs = much lower
        if stats(i).Solidity < 0.7
            abrasion_mask(stats(i).PixelIdxList) = false;
        end
    end
    % ------------------------------------------------------------
    % 5. Debug data
    % ------------------------------------------------------------
    % debug_data.bbox         = [x1 y1 x2-x1+1 y2-y1+1];
    debug_data.threshold    = T;
    % debug_data.gabor_roi    = gabor_roi;
    % debug_data.mask_roi     = mask_roi;
    debug_data.abrasion_mask = abrasion_mask;
    if debug
        figure('Name', 'Abrasion From Precomputed Gabor', 'NumberTitle', 'off');

        subplot(2,3,1);
        imshow(img);
        title('1. Original image');

        subplot(2,3,2);
        imshow(glove_mask);
        title('2. Glove mask');

        % subplot(2,3,3);
        % imshow(img); hold on;
        % rectangle('Position', [x1 y1 x2-x1+1 y2-y1+1], ...
        %           'EdgeColor', 'g', 'LineWidth', 2);
        % title('3. Glove ROI');
        % hold off;

        subplot(2,3,4);
        imagesc(gabor_roi);
        axis image off;
        colormap(gca, 'hot');
        colorbar;
        title('4. Precomputed Gabor ROI');

        subplot(2,3,5);
        imshow(abrasion_roi);
        title(sprintf('5. Thresholded ROI (T=%.3f)', T));

        subplot(2,3,6);
        overlay = img;
        if size(overlay,3) == 1
            overlay = repmat(im2uint8(rescale(overlay)), [1 1 3]);
        end
        if isa(overlay, 'double')
            overlay = im2uint8(overlay);
        end

        r = overlay(:,:,1);
        g = overlay(:,:,2);
        b = overlay(:,:,3);

        r(abrasion_mask) = 255;
        g(abrasion_mask) = uint8(double(g(abrasion_mask)) * 0.4);
        b(abrasion_mask) = uint8(double(b(abrasion_mask)) * 0.4);

        overlay(:,:,1) = r;
        overlay(:,:,2) = g;
        overlay(:,:,3) = b;

        imshow(overlay);
        title('6. Abrasion overlay');

        fprintf('\n--- Abrasion from precomputed Gabor ---\n');
        fprintf('ROI size: %d x %d\n', size(mask_roi,1), size(mask_roi,2));
        fprintf('Threshold: %.4f\n', T);
        fprintf('Detected area: %.2f%% of glove\n', ...
            100 * nnz(abrasion_mask) / max(nnz(glove_mask),1));
    end
end

function drawDefectOverlay(img, masks, ax)
% Draws bounding boxes for all defect masks onto ax.
% Each mask may be a different resolution — boxes are normalised
% back to img dimensions before scaling to axes display size.
%
% INPUTS:
%   img   - original image (used only to get aspect reference)
%   masks - struct with fields: .burn, .Cut, .abrasion
%           each field is a logical mask at whatever resolution
%           that detector ran at
%   ax    - UIAxes handle (image already displayed here)

    [refH, refW, ~] = size(img);
    


    % Display image
    imshow(img, 'Parent', ax);
    
    % Force axes limits to exactly match image pixel dimensions
    ax.XLim = [0.5 refW + 0.5];
    ax.YLim = [0.5 refH + 0.5];
    ax.DataAspectRatio = [1 1 1];

    dispW = diff(ax.XLim);
    dispH = diff(ax.YLim);

    defectTypes = {
        'burn',     'Burn',     [1.0  1.0  0.0];
        'Cut',     'Cut',     [1.0  0.2  0.2];
        'abrasion', 'Abrasion', [1.0  0.55 0.0]
    };

    hold(ax, 'on');

    for d = 1:size(defectTypes, 1)
        fname  = defectTypes{d, 1};
        label  = defectTypes{d, 2};
        colour = defectTypes{d, 3};

        if ~isfield(masks, fname) || isempty(masks.(fname))
            continue;
        end

        mask = logical(masks.(fname));
        if ~any(mask(:))
            continue;
        end

        % This mask may be at a different resolution than img
        [maskH, maskW] = size(mask);

        % Scale factor: mask coords → display coords
        % Goes mask → normalised (0-1) → display pixels
        scaleX = dispW / maskW;
        scaleY = dispH / maskH;

        props = regionprops(mask, 'BoundingBox', 'Centroid');
        if isempty(props)
            continue;
        end

        for k = 1:numel(props)
            bb = props(k).BoundingBox;

            x = bb(1) * scaleX;
            y = bb(2) * scaleY;
            w = bb(3) * scaleX;
            h = bb(4) * scaleY;
            
            % Expand box outward by padding pixels
            padding = 15;  % increase this to make boxes bigger
            x = x - padding;
            y = y - padding;
            w = w + padding * 2;
            h = h + padding * 2;

            rectangle(ax, ...
                'Position',  [x y w h], ...
                'EdgeColor', colour, ...
                'LineWidth', 2);

            text(ax, x, y - 4, ...
                sprintf('%s %d', label, k), ...
                'Color',             colour, ...
                'FontSize',          10, ...
                'FontWeight',        'bold', ...
                'VerticalAlignment', 'bottom');
        end
    end

    hold(ax, 'off');
end

function n = countDefects(mask)
    if isempty(mask) || ~any(mask(:))
        n = 0;
        return;
    end
    n = numel(regionprops(logical(mask), 'Area'));
end