function batch_glove_inspection()
    % =========================================================================
    % GLOVE DEFECT DETECTION SYSTEM (Ultimate Perfect Edition)
    % Targets: Missing Finger, Hole, Loose Thread (Internal & Edge)
    % =========================================================================
    
    % Toggle DEBUG_MODE: set to 'true' for deep single-image analysis, 
    % set to 'false' for the batch dashboard view.
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
    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));
    
    % Apply Gaussian smoothing (reduces high-frequency noise but creates slight edge gaps)
    smoothed_gray = imgaussfilt(gray, 2.5);
    
    % --- STEP 1 & 2: Base Mask Extraction (The Perfect Mask) ---
    level_loose = graythresh(smoothed_gray);
    bw_loose = imbinarize(smoothed_gray, level_loose * 0.8); 
    
    % =========================================================
    % Auto-Polarity Reversal (APR)
    % Dynamically handles dark gloves on bright backgrounds vs. bright gloves on dark backgrounds
    % =========================================================
    % Sample 10x10 pixel regions from the four corners (assumed background)
    corner_pixels = [bw_loose(1:10, 1:10), bw_loose(1:10, end-9:end);
                     bw_loose(end-9:end, 1:10), bw_loose(end-9:end, end-9:end)];
                     
    % If >50% of the corner pixels are white (1), the bright background was misclassified as foreground.
    if sum(corner_pixels(:)) > 0.5 * numel(corner_pixels)
        bw_loose = ~bw_loose; % Invert mask: Ensure the glove remains the logical foreground (1)
    end
    % =========================================================
    
    % Morphological cleanup for the loose mask
    bw_loose = imopen(bw_loose, strel('disk', 3)); 
    bw_loose = bwareafilt(bw_loose, 1); 
    bw_loose = imfill(bw_loose, 'holes'); 
    
    % Generate strict mask and skin mask for internal defect penetration validation
    level_strict = graythresh(smoothed_gray); 
    bw_bright = imbinarize(smoothed_gray, level_strict * 0.9); 
    
    is_skin = (R > G + 10) & (R > B + 10) & (gray > 40); 
    is_skin = imopen(is_skin, strel('disk', 2)); 
    
    bw_strict = bw_bright & ~is_skin & bw_loose;
    bw_strict = imclose(bw_strict, strel('disk', 3)); 
    bw_strict = imopen(bw_strict, strel('disk', 2));
    
    % --- STEP 3: Missing Finger Detection ---
    all_defects = bw_loose & ~bw_strict; 
    clean_defects = bwareafilt(all_defects, [80, 15000]); 
    
    closed_holes_raw = imfill(bw_strict, 'holes') & ~bw_strict;
    closed_holes_raw = bwareaopen(closed_holes_raw, 15); 
    
    missing_fingers = clean_defects & ~closed_holes_raw & is_skin;
    missing_fingers = bwareaopen(missing_fingers, 250);
    
    % =========================================================
    % Fuzzy Border Check (Cropped Wrist Artifact Removal)
    % Strict imclearborder fails due to smoothing gaps. We implement a 10-pixel exclusion zone.
    % =========================================================
    fuzzy_frame_border = false(size(gray));
    % Target the top and bottom 10 rows (typical locations for cropped wrists)
    fuzzy_frame_border(1:10, :) = true; fuzzy_frame_border(end-9:end, :) = true;
    
    % Re-evaluate each missing finger candidate
    possible_missing_fingers_stats = regionprops(missing_fingers, 'PixelIdxList');
    for m = 1:length(possible_missing_fingers_stats)
        object_idx = possible_missing_fingers_stats(m).PixelIdxList;
        % Erase any candidate containing pixels within the fuzzy border zone
        if any(fuzzy_frame_border(object_idx))
            missing_fingers(object_idx) = false; 
        end
    end
    % =========================================================
    
    % Identify genuine dark background pixels
    is_dark_bg = double(gray) < 60; 
    genuine_mask = is_dark_bg | is_skin; 
    
    % =========================================================================
    % STEP 4: Internal Defect Classification (Hole vs. Internal Loose Thread)
    % =========================================================================
    stats_closed_raw = regionprops(closed_holes_raw, 'BoundingBox', 'Area', 'Eccentricity', 'PixelIdxList', 'MinorAxisLength');
    valid_holes = [];
    valid_threads = []; % Array to store all loose threads (internal + edge)
    
    % Color-Agnostic Cuff Detection: 
    % Grayscale gloves have low color variance; colored cuffs (green, blue, etc.) have high variance.
    max_c = max(max(R, G), B);
    min_c = min(min(R, G), B);
    color_diff = max_c - min_c;
    
    % Identify regions with high color variance (>15) that are not skin
    is_cuff = (color_diff > 15) & ~is_skin & bw_loose; 
    is_cuff = bwareaopen(is_cuff, 50);
    
    hole_mask = false(size(gray)); 
    
    for k = 1:length(stats_closed_raw)
        area = stats_closed_raw(k).Area;
        idx = stats_closed_raw(k).PixelIdxList;
        
        % Ignore defects located inside the cuff region
        if sum(is_cuff(idx)) / area > 0.3, continue; end
        
        % Calculate penetration ratio (how much background/skin is visible through the defect)
        genuine_ratio = sum(genuine_mask(idx)) / area;
        
        % Classify as Hole: Large area and high penetration ratio
        if area > 80 && genuine_ratio > 0.15
            valid_holes = [valid_holes; stats_closed_raw(k)];
            hole_mask(idx) = true; 
            
        % Classify as Internal Loose Thread: Medium area, high eccentricity, or low penetration (surface snag)
        elseif area >= 15 && area <= 1000
            if stats_closed_raw(k).Eccentricity > 0.70 || genuine_ratio <= 0.15
                valid_threads = [valid_threads; stats_closed_raw(k)];
            end
        end
    end
    
    % --- Exclusion Zone Mask ---
    % Create a 20-pixel border exclusion zone to prevent edge artifacts
    border_mask = false(size(gray));
    border_mask(1:20, :) = true; border_mask(end-19:end, :) = true;
    border_mask(:, 1:20) = true; border_mask(:, end-19:end) = true;
    
    % Dilate verified defects and cuffs to create a buffer zone against false edge threads
    exclusion_zone = imdilate(missing_fingers | hole_mask | is_cuff, strel('disk', 25)) | border_mask;
    
    % =========================================================================
    % STEP 5: Edge Loose Thread Detection (Silhouette Subtraction Method)
    % =========================================================================
    % Erode protrusions from the silhouette using a large disk
    bw_solid = imopen(bw_loose, strel('disk', 6)); 
    
    % Subtract the solid silhouette from the raw silhouette to isolate edge threads
    edge_raw_threads = bw_loose & ~bw_solid;
    
    % Filter out micro-noise and apply the exclusion zone
    edge_raw_threads = bwareaopen(edge_raw_threads, 10); 
    edge_raw_threads = edge_raw_threads & ~exclusion_zone;
    
    valid_edge_threads_mask = false(size(gray)); 
    stats_edge = regionprops(edge_raw_threads, 'BoundingBox', 'Area', 'Eccentricity', 'PixelIdxList', 'MinorAxisLength');
    
    for k = 1:length(stats_edge)
        idx = stats_edge(k).PixelIdxList;
        mean_intensity = mean(double(gray(idx))); 
        
        area = stats_edge(k).Area;
        ecc = stats_edge(k).Eccentricity;
        minor_axis = stats_edge(k).MinorAxisLength;
        
        % Rigorous dual-channel geometric verification:
        % Channel 1: Thin and straight fibers
        is_thin_straight = (area > 20) && (ecc > 0.80) && (minor_axis < 10) && (mean_intensity > graythresh(smoothed_gray) * 255 * 0.60);
        
        % Channel 2: Thick, curly tangles (requires higher intensity to exclude shadows)
        is_curly_bright = (area > 40) && (minor_axis < 15) && (mean_intensity > graythresh(smoothed_gray) * 255 * 0.75);
        
        if is_thin_straight || is_curly_bright
            valid_threads = [valid_threads; stats_edge(k)]; 
            if is_debug
                valid_edge_threads_mask(idx) = true; 
            end
        end
    end
    
    % =========================================================================
    % DEBUG MODE: 6-Panel Visualization
    % =========================================================================
    if is_debug
        figure('Name', ['Debug Viewer: ', filename], 'Position', [100, 100, 1400, 800]);
        
        subplot(2,3,1); imshow(img); 
        title('1. Original Image', 'FontSize', 12);
        
        subplot(2,3,2); imshow(bw_loose); 
        title('2. Glove Body Mask', 'FontSize', 12);
        
        subplot(2,3,3); imshow(genuine_mask); 
        title('3. Penetration Validation (Skin & Dark BG)', 'FontSize', 12, 'Color', 'g');
        
        subplot(2,3,4); imshow(missing_fingers); 
        title('4. Missing Finger Detection', 'FontSize', 12, 'Color', 'm');
        
        subplot(2,3,5); imshow(exclusion_zone); 
        title('5. Spatial Exclusion Zone', 'FontSize', 12, 'Color', 'r');
        
        subplot(2,3,6); imshow(valid_edge_threads_mask); 
        title('6. Edge Threads Extraction', 'FontSize', 12, 'Color', 'y');
        
        sgtitle(['Deep Debug Mode - ', filename], 'FontSize', 16, 'FontWeight', 'bold');
        return; 
    end
    
    % =========================================================================
    % Final Rendering & Bounding Box Annotation
    % =========================================================================
    imshow(img); hold on;
    defect_found = false;
    pad = 10; 
    fs_large = 10; fs_mid = 9; fs_small = 8;
    
    % Draw Missing Finger bounding boxes
    stats_missing = regionprops(missing_fingers, 'BoundingBox');
    for k = 1:length(stats_missing)
        bbox = add_padding(stats_missing(k).BoundingBox, pad, imgWidth, imgHeight);
        rectangle('Position', bbox, 'EdgeColor', 'm', 'LineWidth', 2.5);
        text(bbox(1), bbox(2)-8, 'Missing Finger', 'Color', 'm', 'FontWeight', 'bold', 'FontSize', fs_large);
        defect_found = true;
    end
    
    % Draw Hole bounding boxes
    for k = 1:length(valid_holes)
        bbox = add_padding(valid_holes(k).BoundingBox, pad, imgWidth, imgHeight);
        rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
        text(bbox(1), bbox(2)-8, 'Hole', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', fs_mid);
        defect_found = true;
    end
    
    % Draw Loose Thread bounding boxes
    for k = 1:length(valid_threads)
        bbox = add_padding(valid_threads(k).BoundingBox, pad, imgWidth, imgHeight);
        rectangle('Position', bbox, 'EdgeColor', 'y', 'LineWidth', 1.5);
        text(bbox(1), bbox(2)-8, 'Loose Thread', 'Color', 'y', 'FontWeight', 'bold', 'FontSize', fs_small);
        defect_found = true;
    end
    
    % Render Title (PASS/DEFECTIVE)
    [~, short_name, ~] = fileparts(filename);
    if ~defect_found
        title(sprintf('%s: PASS', short_name), 'Color', 'g', 'FontSize', 12, 'Interpreter', 'none');
    else
        title(sprintf('%s: DEFECTIVE', short_name), 'Color', 'r', 'FontSize', 12, 'Interpreter', 'none');
    end
    hold off;
end

% Helper function to apply padding to bounding boxes safely
function new_bbox = add_padding(bbox, pad, imgW, imgH)
    x = max(1, bbox(1) - pad);
    y = max(1, bbox(2) - pad);
    w = min(imgW - x, bbox(3) + 2*pad);
    h = min(imgH - y, bbox(4) + 2*pad);
    new_bbox = [x, y, w, h];
end