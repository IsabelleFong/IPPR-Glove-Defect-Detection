%% ====================== Nitrile Glove Main Function ======================
function [result, defect] = detectNitrileGloveDefect(img, appUI)

    result = '';
    defect = '';

    % Resize Image
    [img, success] = resizeImage(img);

    if ~success
        fprintf('Image failed to resize.\n');
        return;
    else 
        fprintf('Image resized successfully.\n');
    end

    % Pre-process the Image
    [gray_img, success] = imagePreprocessing(img);

    if ~success
        fprintf('Image preprocessing failed. Exiting.\n');
        return;
    else
        fprintf('Image preprocessesd successfully.\n');
    end

    % Skin Mask
    skin_mask = skinExtraction(img);

    % Extract Glove
    [extracted_glove, glove_mask] = gloveExtraction(img, gray_img, skin_mask);

    if ~success
        fprintf('Glove extraction failed. Exiting.\n');
        return;
    else
        fprintf('Glove extracted successfully.\n');
    end

    % Intensity Mapping
    [intensity_map, intensity_deviation_mask] = findIntensityDeviation(img, glove_mask);

    % Texture Mapping
    [texture_map, texture_deviation_mask] = findTextureDeviation(img, glove_mask);

    candidate_defects = intensity_deviation_mask | texture_deviation_mask;
    
    [defect_class_map, defect] = classifyGloveDefects(candidate_defects, intensity_deviation_mask, texture_deviation_mask, glove_mask, skin_mask);

    if all(defect_class_map(:) == 0)
        result = 'NON-DEFECTIVE';
        fprintf('RESULT: NON-DEFECTIVE (No Defects Found)\n');
    else
        drawBoundingBoxes(defect_class_map, img, appUI);
        result = 'DEFECTIVE';
        fprintf('RESULT: DEFECTIVE (Defect Found)\n');
    end

end


%% ====================== Resize Image Function ======================
function [img, success] = resizeImage(img)

    try
        % Set target size
        target_size = [4000, 3000];  % [rows, cols]
        
        % Resize original image
        img = imresize(img, target_size);
        success = true;
    catch
        img = [];
        success = false;
    end

end


%% ====================== Image Pre-processing Function ======================
function [gray_img, success] = imagePreprocessing(img)
    try 
        % STEP 1: Convert to Grayscale and Darken
        gray_img = im2gray(img);      
        gray_img = im2double(gray_img);
        gray_img = gray_img * 0.5; 
        gray_img = im2uint8(gray_img);   

        % STEP 2: Remove Shadows / Uneven Illumination
        background = imopen(gray_img, strel('disk', 50));
        gray_img = imsubtract(gray_img, background);

        background = imopen(gray_img, strel('disk', 50));
        gray_img = imsubtract(gray_img, background);

        % STEP 3: Blur / Reduce Sharpness
        sigma = 10;                          % Standard deviation for Gaussian blur
        gray_img = imgaussfilt(gray_img, sigma);

        % STEP 4: Enhance Contrast
        gray_img = adapthisteq(gray_img, 'ClipLimit', 0.03);

        success = true;

    catch 
        gray_img = [];
        success = false;
    end
end


%% ====================== Extract Glove Function ======================
function [extracted_glove, glove_mask] = gloveExtraction(img, gray_img, skin_mask)

   % STEP 1: Create Glove Mask
   T = graythresh(gray_img);
   bw = imbinarize(gray_img, T); 

   % Remove small objects and smooth mask
    bw = bwareaopen(bw, 50);
    bw = imfill(bw, 'holes');
    glove_mask = imclose(bw, strel('disk', 200));

    % Keep only largest connected component
    cc = bwconncomp(glove_mask);
    numPixels = cellfun(@numel, cc.PixelIdxList);
    [~, idx] = max(numPixels);
    largest_glove_mask = false(size(glove_mask));
    largest_glove_mask(cc.PixelIdxList{idx}) = true;
    glove_mask = largest_glove_mask;
    glove_mask = imfill(glove_mask, 'holes');

    % Remove skin from glove mask
    glove_mask(skin_mask) = 0;

    % STEP 2: Extract the glove from the original image using the mask
    extracted_glove = img;
    
    % Get number of channels
    numChannels = size(img, 3);
    
    % Apply mask to each channel
    for c = 1:numChannels
        temp = extracted_glove(:,:,c);
        temp(~glove_mask) = 0;  % set background to black
        extracted_glove(:,:,c) = temp;
    end

end


%% ========================= Extract Skin in Glove Function ===============================
function skin_mask = skinExtraction(img)
    hsv = rgb2hsv(img);
    H = hsv(:,:,1); S = hsv(:,:,2); V = hsv(:,:,3);
    
    ycbcr = rgb2ycbcr(img);
    Cb = ycbcr(:,:,2);
    Cr = ycbcr(:,:,3);

    skin_hsv = (H > 0 & H < 0.1) & (S > 0.2 & S < 0.7) & (V > 0.4 & V < 1);
    skin_ycbcr = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173);
    
    skin_mask = skin_hsv & skin_ycbcr;
end

%% ======================== Intensity Deviation Function ========================
function [intensity_map, deviation_mask] = findIntensityDeviation(img, glove_mask)
    % STEP 1: Convert to Grayscale
    gray_img = im2gray(img);
    gray_img(~glove_mask) = 0;

    % Set threshold deivation
    threshold = 120;

    % Ensure image is double for calculations
    gray_double = double(gray_img);

    % Extract only glove pixels
    glove_pixels = gray_double .* double(glove_mask);

    % Compute mean intensity of glove
    mean_intensity = mean(glove_pixels(glove_mask));

    % Compute deviation from mean
    deviation = abs(glove_pixels - mean_intensity);

    % Detect pixels exceeding threshold
    deviation_mask = false(size(gray_img));
    deviation_mask(deviation > threshold & glove_mask) = true;

    % Create intensity map for visualization
    intensity_map = zeros(size(gray_img));
    intensity_map(glove_mask) = gray_double(glove_mask);
end


%% ======================== Texture Deviation Function (Std-based) ========================
function [texture_map, deviation_mask] = findTextureDeviation(img, glove_mask)
    % STEP 1: Convert to Grayscale
    gray_img = im2gray(img);
    gray_double = double(gray_img);

    % Apply glove mask to zero-out background
    gray_double(~glove_mask) = 0;

    % STEP 2: Compute local texture map using local standard deviation
    window_size = 7;  
    local_std = stdfilt(gray_double, true(window_size));

    % Mask out non-glove pixels
    texture_map = zeros(size(gray_img));
    texture_map(glove_mask) = local_std(glove_mask);

    % STEP 3: Compute global standard deviation of glove texture
    std_texture = std(local_std(glove_mask));

    % STEP 4: Detect deviations purely based on standard deviation
    deviation_mask = false(size(gray_img));
    % Flag pixels exceeding n times the glove's texture std
    deviation_factor = 2.5;  % adjust as needed
    deviation_mask(glove_mask) = local_std(glove_mask) > deviation_factor * std_texture;
end

%% ======================== Detect Stains =======================================

function stain_mask = detectStain(candidate_mask, intensity_mask)

    cc = bwconncomp(candidate_mask);
    stats = regionprops(cc, 'Area', 'Eccentricity', ...
                             'Solidity', 'PixelIdxList');

    stain_mask = false(size(candidate_mask));

    % ---------------- Mark candidate regions ----------------
    for i = 1:length(stats)
        if stats(i).Area > 100 && stats(i).Solidity > 0.1
            % Must also have intensity deviation
            if any(intensity_mask(stats(i).PixelIdxList))
                stain_mask(stats(i).PixelIdxList) = true;
            end
        end
    end


    % ---------------- Merge nearby stain regions ----------------
    se = strel('disk', 100);   % adjust the radius based on distance to merge
    stain_mask = imclose(stain_mask, se);


    % ---------------- Keep only the largest region if large enough ----------------
    cc_merged = bwconncomp(stain_mask);
    if cc_merged.NumObjects > 0
        stats_merged = regionprops(cc_merged, 'Area', 'PixelIdxList');
        % Find largest
        [max_area, idx] = max([stats_merged.Area]);
        if max_area > 6000
            largest_mask = false(size(stain_mask));
            largest_mask(stats_merged(idx).PixelIdxList) = true;
            stain_mask = largest_mask;
        else
            stain_mask = false(size(stain_mask)); % no region large enough
        end
    end

    % Option 1: using sum
    numPixels = sum(stain_mask(:));
    fprintf('The stain region has %d pixels\n', numPixels);
    
    % Option 2: using nnz (number of non-zero elements)
    numPixels = nnz(stain_mask);
    fprintf('The stain region has %d pixels\n', numPixels);

end



%% ======================== Detect Folds ====================================

function fold_mask = detectFold(candidate_mask, texture_mask, stain_mask)
    % window_pct: percentage of image dimensions for window (e.g., 0.1 = 10%)
    window_pct = 0.2;

    % step_pct: percentage of image dimensions for step (e.g., 0.02 = 2%)
    step_pct = 0.01;

    % ---------------- Step 0: Restrict texture_mask to candidate_mask ----------------
    candidate_texture = false(size(texture_mask));
    candidate_texture(candidate_mask) = texture_mask(candidate_mask);

    % Remove stain regions
    candidate_texture(stain_mask) = false;

    % ---------------- Step 1: Merge nearby candidate pixels ----------------

    candidate_texture = imopen(candidate_texture, strel('disk', 5));

    merged_texture = imclose(candidate_texture, strel('disk', 50));

    % ---------------- Step 2: Sliding window density check ----------------
    [rows, cols] = size(merged_texture);

    % Compute window size and step size in pixels
    window_rows = max(1, round(rows * window_pct));
    window_cols = max(1, round(cols * window_pct));
    step_rows = max(1, round(rows * step_pct));
    step_cols = max(1, round(cols * step_pct));

    max_ratio = 0;
    best_window = [];

    for r = 1:step_rows:(rows - window_rows + 1)
        for c = 1:step_cols:(cols - window_cols + 1)
            patch = merged_texture(r:r+window_rows-1, c:c+window_cols-1);
            ratio = nnz(patch) / numel(patch);  % ratio of 1s
            if ratio > max_ratio
                max_ratio = ratio;
                best_window = [r, c];  % top-left of densest patch
            end
        end
    end

    % ---------------- Step 3: Create fold_mask using best patch ----------------
    fold_mask = false(size(merged_texture));
    if ~isempty(best_window)
        r = best_window(1);
        c = best_window(2);
        fold_mask(r:r+window_rows-1, c:c+window_cols-1) = ...
            merged_texture(r:r+window_rows-1, c:c+window_cols-1);
    end

    % ---------------- Step 4: Check area ----------------
    if nnz(fold_mask) < 5000
        fold_mask = false(size(fold_mask));  % area too small → discard
        fprintf('Fold mask discarded: area < 5000 pixels\n');
    else
        fprintf('Fold mask accepted: area = %d pixels\n', nnz(fold_mask));
    end

end


%% ======================== Detect Tear =================================
function tear_mask = detectTear(skin_mask, glove_mask)

    filled_glove_mask = imfill(glove_mask, 'holes');
    
    % Extract skin inside the glove
    skin_inside_glove = skin_mask & filled_glove_mask;
    
    % Remove very small regions
    skin_inside_glove = bwareaopen(skin_inside_glove, 50);
    
    % Smooth / merge regions
    tear_mask = imclose(skin_inside_glove, strel('disk', 50));
    
    % Keep Largest Connected Region if it is big enough
    minArea = 2000;  % minimum number of pixels to consider a valid tear
    cc = bwconncomp(tear_mask);
    
    if cc.NumObjects > 0
        stats = regionprops(cc, 'Area', 'PixelIdxList');
        % Find the largest area
        [maxArea, idx] = max([stats.Area]);
        
        if maxArea >= minArea
            % Keep the largest region
            largest_region = false(size(tear_mask));
            largest_region(stats(idx).PixelIdxList) = true;
            tear_mask = largest_region;
        else
            % Largest region too small → return empty mask
            tear_mask = false(size(tear_mask));
        end
    else
        tear_mask = false(size(tear_mask));
    end

    % Option 1: using sum
    numPixels = sum(tear_mask(:));
    fprintf('The tear region has %d pixels\n', numPixels);
    
    % Option 2: using nnz (number of non-zero elements)
    numPixels = nnz(tear_mask);
    fprintf('The tear region has %d pixels\n', numPixels);

end

%% ======================== Defect Classification Function ========================

function [defect_class_map, defect] = classifyGloveDefects(candidate_mask, intensity_mask, texture_mask, glove_mask, skin_mask)

    % Initialize output map
    defect_class_map = zeros(size(glove_mask));

    % Detect Tears
    tear_mask = detectTear(skin_mask, glove_mask);

    defect = '';

    if any(tear_mask(:))
        defect = 'Tear';
        fprintf('Defect (Tear) detected.\n');
        defect_class_map(tear_mask)  = 3; % Green
        return;
    end

    % Detect Stains
    stain_mask = detectStain(candidate_mask, intensity_mask);

    if any(stain_mask(:))
        defect = 'Stain';
        fprintf('Defect (Stain) detected.\n');
        defect_class_map(stain_mask)  = 1; % Red
        return;
    end

    % Detect Folds
    fold_mask = detectFold(candidate_mask, texture_mask, stain_mask);

    if any(fold_mask(:))
        defect = 'Fold';
        fprintf('Defect (Fold) detected.\n');
        defect_class_map(fold_mask)  = 2; % Blue
        return;
    end

end

%% ====================== Draw Bounding Boxes ==============================
function drawBoundingBoxes(defect_class_map, img, appUI)
    % ------ Merge Bounding Boxes by Defect Type with Proximity Margin -----

    proximityMargin = 100;

    % Get connected components
    cc = bwconncomp(defect_class_map > 0);
    stats = regionprops(cc, 'BoundingBox', 'PixelIdxList');

    % Extract all bounding boxes and their classes
    allBoxes = cat(1, stats.BoundingBox);  % Nx4: [x y w h]
    boxesXY = [allBoxes(:,1), allBoxes(:,2), allBoxes(:,1)+allBoxes(:,3), allBoxes(:,2)+allBoxes(:,4)];

    % Determine class for each box
    boxClasses = zeros(length(stats),1);
    for k = 1:length(stats)
        boxClasses(k) = mode(defect_class_map(stats(k).PixelIdxList));
    end

    % -------------------- Merge Boxes by Class ----------------------------

    merged = true;
    while merged
        merged = false;
        N = size(boxesXY,1);
        i = 1;
        while i <= N
            j = i + 1;
            while j <= N
                % Only merge boxes of the same class
                if boxClasses(i) == boxClasses(j)
                    % Expand boxes by proximity margin
                    box1 = boxesXY(i,:) + [-proximityMargin -proximityMargin proximityMargin proximityMargin];
                    box2 = boxesXY(j,:) + [-proximityMargin -proximityMargin proximityMargin proximityMargin];

                    % Check overlap with expanded boxes
                    if ~(box1(3)<box2(1) || box1(1)>box2(3) || box1(4)<box2(2) || box1(2)>box2(4))
                        % Merge original boxes
                        newBox = [min(boxesXY(i,1), boxesXY(j,1)), ...
                                  min(boxesXY(i,2), boxesXY(j,2)), ...
                                  max(boxesXY(i,3), boxesXY(j,3)), ...
                                  max(boxesXY(i,4), boxesXY(j,4))];
                        boxesXY(i,:) = newBox;

                        % Keep class the same
                        boxClasses(i) = boxClasses(i);

                        % Remove merged box
                        boxesXY(j,:) = [];
                        boxClasses(j) = [];
                        merged = true;
                        N = N - 1;
                    else
                        j = j + 1;
                    end
                else
                    j = j + 1;
                end
            end
            i = i + 1;
        end
    end

    % -------------------------- Draw Merged Boxes -------------------------

    % Display image in the app's UIAxes
    imshow(img, 'Parent', appUI);  % show image in app
    hold(appUI, 'on');             % allow drawing on top
    % title(app.UI, 'Defect Detected');
    
    for i = 1:size(boxesXY,1)
        x = boxesXY(i,1);
        y = boxesXY(i,2);
        w = boxesXY(i,3) - x;
        h = boxesXY(i,4) - y;
    
        % Assign label and color based on class - set to all the same colour
        switch boxClasses(i)
            case 1, label='Stain'; boxColor='r';
            case 2, label='Fold';  boxColor='r';
            case 3, label='Tear';  boxColor='r';
        end
    
        % Draw rectangle on the UIAxes
        rectangle(appUI, 'Position', [x y w h], 'EdgeColor', boxColor, 'LineWidth', 2);
    
        % Draw label text above rectangle
        text(appUI, x, y-5, label, ...
            'Color', boxColor, ...
            'FontSize', 12, ...
            'FontWeight', 'bold', ...
            'BackgroundColor', 'black');
    end
    
    hold(appUI, 'off');  % release hold

end