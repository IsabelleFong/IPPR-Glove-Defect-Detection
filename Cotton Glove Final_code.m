function batch_glove_inspection()
    % =========================================================================
    % GLOVE DEFECT DETECTION SYSTEM (Ultimate V3 - Global Skin & Anti-Vertcat)
    % Targets: Missing Finger, Hole, Loose Thread (Internal & Edge)
    % =========================================================================
    
    % Toggle DEBUG_MODE: 'true' for deep 6-panel analysis of a single image, 
    % 'false' for the automated batch dashboard view.
    DEBUG_MODE = false; 
    
    if DEBUG_MODE
        disp('--- DEBUG MODE ACTIVE: Recommend selecting 1 image for deep analysis ---');
    end
    
    % Prompt user to select image files
    [files, path] = uigetfile({'*.png;*.jpg;*.bmp', 'Image Files'}, ...
        'Select Glove Images', 'MultiSelect', 'on');
    
    if isequal(files, 0), return; end
    if ischar(files), files = {files}; end
    num_images = length(files);
    
    % Initialize the batch dashboard figure if not in debug mode
    if ~DEBUG_MODE
        figure('Name', 'AOI Batch Glove Inspection Dashboard', 'Position', [50, 50, 1600, 900]);
        sgtitle(sprintf('Automated Optical Inspection (AOI) - Batch Results (%d Images)', num_images), ...
            'FontSize', 18, 'FontWeight', 'bold', 'Color', 'b');
        cols = ceil(sqrt(num_images));
        rows = ceil(num_images / cols);
        if num_images == 12, rows = 3; cols = 4; end
    end
    
    % Loop through selected images
    for i = 1:num_images
        img_path = fullfile(path, files{i});
        img = imread(img_path);
        if ~DEBUG_MODE, subplot(rows, cols, i); end
        process_single_glove(img, files{i}, DEBUG_MODE);
    end
end

% =========================================================================
% Core Algorithm Module
% =========================================================================
function process_single_glove(img, filename, is_debug)
    [imgHeight, imgWidth, ~] = size(img);
    gray = rgb2gray(img);
    R = double(img(:,:,1)); G = double(img(:,:,2)); B = double(img(:,:,3));
    
    % Apply Gaussian smoothing to reduce high-frequency noise
    smoothed_gray = imgaussfilt(gray, 2.5);
    
    % --- STEP 1 & 2: Base Mask Extraction & Auto-Polarity Reversal (APR) ---
    level_loose = graythresh(smoothed_gray);
    bw_loose = imbinarize(smoothed_gray, level_loose * 0.8); 
    
    % APR: Sample corner pixels to determine background color dynamically
    corner_pixels = [bw_loose(1:10, 1:10), bw_loose(1:10, end-9:end);
                     bw_loose(end-9:end, 1:10), bw_loose(end-9:end, end-9:end)];
    if sum(corner_pixels(:)) > 0.5 * numel(corner_pixels)
        bw_loose = ~bw_loose; % Invert mask if background is incorrectly classified as foreground
    end
    
    % Morphological cleanup
    bw_loose = imopen(bw_loose, strel('disk', 3)); 
    bw_loose = bwareafilt(bw_loose, 1); 
    bw_loose = imfill(bw_loose, 'holes'); 
    
    % --- 🌟 STEP 3: Global Skin Detection (YCbCr Space) ---
    % Convert to YCbCr space for robust, illumination-invariant skin detection
    ycbcr = rgb2ycbcr(img);
    Cr = ycbcr(:,:,3);
    Cb = ycbcr(:,:,2);
    
    % Industrial-standard chrominance thresholds to accommodate diverse human skin tones
    is_skin = (Cr > 133) & (Cr < 173) & (Cb > 77) & (Cb < 127);
    is_skin = imopen(is_skin, strel('disk', 2)); 
    
    % Generate strict mask based on brightness and skin exclusion
    level_strict = graythresh(smoothed_gray); 
    bw_bright = imbinarize(smoothed_gray, level_strict * 0.9); 
    bw_strict = imopen(imclose(bw_bright & ~is_skin & bw_loose, strel('disk', 3)), strel('disk', 2));
    
    % --- Missing Finger Detection ---
    % Detect regions missing from the strict mask that reveal skin
    missing_fingers = bwareaopen((bw_loose & ~bw_strict) & ~bwareaopen(imfill(bw_strict, 'holes') & ~bw_strict, 15) & is_skin, 250);
    
    % Define top and bottom 10-pixel border zones to prevent cropped wrists from being flagged as missing fingers
    fuzzy_frame_border = false(size(gray));
    fuzzy_frame_border(1:10, :) = true; fuzzy_frame_border(end-9:end, :) = true;
    
    mf_stats = regionprops(missing_fingers, 'PixelIdxList', 'BoundingBox');
    
    % 🛡️ Use Cell Arrays to store coordinates, completely eliminating 'vertcat' structure mismatch errors
    final_missing_boxes = {}; 
    for m = 1:length(mf_stats)
        if any(fuzzy_frame_border(mf_stats(m).PixelIdxList))
            missing_fingers(mf_stats(m).PixelIdxList) = false; 
        else
            final_missing_boxes{end+1} = mf_stats(m).BoundingBox;
        end
    end
    
    % =========================================================
    % 🛡️ Unified Border Exclusion Mask (Shared by Internal & Edge Steps)
    % =========================================================
    border_mask = false(size(gray));
    border_mask(1:20, :) = true;           
    % 🚨 Crucial Fix: Extended bottom exclusion zone to 60 pixels to suppress ribbed wrist fuzz artifacts
    border_mask(end-60:end, :) = true;     
    border_mask(:, 1:20) = true;           
    border_mask(:, end-19:end) = true;     
    
    % --- STEP 4: Internal Defect Classification (Holes & Internal Threads) ---
    is_dark_bg = double(gray) < 60; 
    genuine_mask = is_dark_bg | is_skin; % Valid background penetration (either dark table or skin)
    
    closed_holes_raw = bwareaopen(imfill(bw_strict, 'holes') & ~bw_strict, 15);
    stats_closed = regionprops(closed_holes_raw, 'BoundingBox', 'Area', 'Eccentricity', 'PixelIdxList');
    
    % Color-agnostic cuff detection to prevent colored cuffs from being misclassified as holes
    color_diff = max(max(R, G), B) - min(min(R, G), B);
    is_cuff = bwareaopen((color_diff > 15) & ~is_skin & bw_loose, 50);
    
    final_hole_boxes = {};
    final_thread_boxes = {}; 
    hole_mask = false(size(gray)); 
    
    for k = 1:length(stats_closed)
        area = stats_closed(k).Area;
        idx = stats_closed(k).PixelIdxList;
        
        if any(border_mask(idx)), continue; end % Ignore internal holes touching the rough edges
        if sum(is_cuff(idx)) / area > 0.3, continue; end
        
        % Calculate how much true background is visible through the defect
        gen_ratio = sum(genuine_mask(idx)) / area;
        
        if area > 80 && gen_ratio > 0.15
            final_hole_boxes{end+1} = stats_closed(k).BoundingBox;
            hole_mask(idx) = true; 
        elseif area >= 15 && area <= 1000
            % 🚨 Core Fix: Internal shadows must be extremely elongated (Eccentricity > 0.88) to be considered loose threads
            if stats_closed(k).Eccentricity > 0.88 
                final_thread_boxes{end+1} = stats_closed(k).BoundingBox;
            end
        end
    end
    
    % --- STEP 5: Edge Loose Thread Detection (Pure Geometric Analysis) ---
    % Erode the loose mask to create a solid silhouette
    bw_solid = imopen(bw_loose, strel('disk', 6)); 
    edge_raw_threads = bw_loose & ~bw_solid; % Subtract solid mask from loose mask to isolate protrusions
    
    % Surgical Exclusion: 25px radius around missing fingers/cuffs, but only 8px around holes to preserve nearby threads
    exclusion_zone = imdilate(missing_fingers | is_cuff, strel('disk', 25)) | imdilate(hole_mask, strel('disk', 8)) | border_mask;
    
    % Filter micro-noise and apply exclusion zone
    edge_raw_threads = bwareaopen(edge_raw_threads, 20) & ~exclusion_zone;
    stats_edge = regionprops(edge_raw_threads, 'BoundingBox', 'Area', 'Eccentricity', 'PixelIdxList', 'MinorAxisLength');
    
    valid_edge_threads_mask = false(size(gray)); 
    
    for k = 1:length(stats_edge)
        area = stats_edge(k).Area;
        ecc = stats_edge(k).Eccentricity;
        minor = stats_edge(k).MinorAxisLength;
        
        % Purely geometric filtering (removed problematic intensity checks)
        % Criteria 1: Thin and straight fibers (higher area threshold to reject cotton fuzz)
        is_thin_thread = (area > 30) && (ecc > 0.88) && (minor < 12);
        
        % Criteria 2: Thicker, curly fiber tangles
        is_curly_thread = (area > 50) && (minor < 18);
        
        if is_thin_thread || is_curly_thread
            final_thread_boxes{end+1} = stats_edge(k).BoundingBox;
            if is_debug, valid_edge_threads_mask(stats_edge(k).PixelIdxList) = true; end
        end
    end
    
    % =========================================================================
    % DEBUG MODE & FINAL RENDERING
    % =========================================================================
    if is_debug
        figure('Name', ['Debug Viewer: ', filename], 'Position', [100, 100, 1400, 800]);
        subplot(2,3,1); imshow(img); title('1. Original Image', 'FontSize', 12);
        subplot(2,3,2); imshow(bw_loose); title('2. Glove Body Mask', 'FontSize', 12);
        subplot(2,3,3); imshow(genuine_mask); title('3. Global Penetration Valid', 'FontSize', 12, 'Color', 'g');
        subplot(2,3,4); imshow(missing_fingers); title('4. Missing Finger Layer', 'FontSize', 12, 'Color', 'm');
        subplot(2,3,5); imshow(exclusion_zone); title('5. Surgical Exclusion Zone', 'FontSize', 12, 'Color', 'r');
        subplot(2,3,6); imshow(valid_edge_threads_mask); title('6. Processed Edge Threads', 'FontSize', 12, 'Color', 'y');
        sgtitle(['Deep Debug Mode - ', filename], 'FontSize', 16, 'FontWeight', 'bold');
        return; 
    end
    
    imshow(img); hold on;
    detected_names = {};
    pad = 10; fs_large = 10; fs_mid = 9; fs_small = 8;
    
    % Draw Missing Finger (Magenta)
    if ~isempty(final_missing_boxes)
        detected_names{end+1} = 'Missing Finger';
        for k = 1:length(final_missing_boxes)
            bbox = add_padding(final_missing_boxes{k}, pad, imgWidth, imgHeight);
            rectangle('Position', bbox, 'EdgeColor', 'm', 'LineWidth', 2.5);
            text(bbox(1), bbox(2)-8, 'Missing Finger', 'Color', 'm', 'FontWeight', 'bold', 'FontSize', fs_large);
        end
    end
    
    % Draw Hole (Red)
    if ~isempty(final_hole_boxes)
        detected_names{end+1} = 'Hole';
        for k = 1:length(final_hole_boxes)
            bbox = add_padding(final_hole_boxes{k}, pad, imgWidth, imgHeight);
            rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
            text(bbox(1), bbox(2)-8, 'Hole', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', fs_mid);
        end
    end
    
    % Draw Loose Thread (Yellow)
    if ~isempty(final_thread_boxes)
        detected_names{end+1} = 'Loose Thread';
        for k = 1:length(final_thread_boxes)
            bbox = add_padding(final_thread_boxes{k}, pad, imgWidth, imgHeight);
            rectangle('Position', bbox, 'EdgeColor', 'y', 'LineWidth', 1.5);
            text(bbox(1), bbox(2)-8, 'Loose Thread', 'Color', 'y', 'FontWeight', 'bold', 'FontSize', fs_small);
        end
    end
    
    % Dynamic Title Rendering (Concatenating multiple defects)
    [~, short_name, ~] = fileparts(filename);
    if isempty(detected_names)
        title(sprintf('%s: PASS', short_name), 'Color', 'g', 'FontSize', 12, 'Interpreter', 'none');
    else
        defect_str = strjoin(unique(detected_names), ', ');
        title(sprintf('%s: %s', short_name, defect_str), 'Color', 'r', 'FontSize', 12, 'Interpreter', 'none');
    end
    hold off;
end

% Helper function to safely expand the bounding box to prevent cropping at image edges
function new_bbox = add_padding(bbox, pad, imgW, imgH)
    x = max(1, bbox(1) - pad);
    y = max(1, bbox(2) - pad);
    w = min(imgW - x, bbox(3) + 2*pad);
    h = min(imgH - y, bbox(4) + 2*pad);
    new_bbox = [x, y, w, h];
end
