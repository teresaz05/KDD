function reducedVec = reducePrecision(vec, exponentBits, mantissaBits)
    reducedVec = zeros(size(vec), 'single');

    % Sanity check: Ensure bit counts are within valid IEEE-754 single-precision range
    if (exponentBits > 8) || (mantissaBits > 23)
        error('Exponent or mantissa is out of range'); 
    end

    % Bitmasks for extracting fields from IEEE-754 32-bit representation
    mask511 = uint32(ones(size(vec)) * 511);                % 9-bit mask
    mask255 = uint32(ones(size(vec)) * 255);                % 8-bit mask
    mask2_23minus1 = uint32(ones(size(vec)) * (2^23 - 1));  % Mask for 23 mantissa bits
    mantissaMask = uint32(ones(size(vec)) * ...
                     bitshift(2^mantissaBits - 1, 23 - mantissaBits)); % Mask to retain desired mantissa bits

    if (exponentBits == 8)
        % Case 1: Full 8-bit exponent is retained
        numBits = typecast(vec, 'uint32');  % Reinterpret float as uint32
        signANDexponent = bitand(bitshift(numBits, -23), mask511);  % Extract sign + exponent
        mantissa = bitand(numBits, mask2_23minus1);  % Extract mantissa
        reducedMantissa = bitand(mantissa, mantissaMask);  % Truncate mantissa bits
        reducedNumBits = bitor(bitshift(signANDexponent, 23), reducedMantissa);  % Reconstruct bits
        reducedVec = typecast(uint32(reducedNumBits), 'single');  % Convert back to float

    else
        % Case 2: Exponent needs to be truncated (fewer than 8 bits)
        reducedexponentmax = uint32(ones(size(vec)) * (128 + 2^(exponentBits - 1)));  % Max allowed exponent
        reducedexponentmin = uint32(ones(size(vec)) * (128 - 2^(exponentBits - 1)));  % Min allowed exponent

        numBits = typecast(vec, 'uint32');  % Reinterpret float as uint32
        signBit = bitshift(numBits, -31);  % Extract sign bit
        exponent = bitand(bitshift(numBits, -23), mask255);  % Extract original exponent
        mantissa = bitand(numBits, 2^23 - 1);  % Extract original mantissa

        % Clamp exponent to reduced bit range
        reducedexponent = min(max(exponent, reducedexponentmin), reducedexponentmax);
        reducedMantissa = bitand(mantissa, mantissaMask);  % Truncate mantissa bits

        % Rebuild float with reduced precision
        reducedNumBits = bitor(bitor(bitshift(signBit, 31), bitshift(reducedexponent, 23)),reducedMantissa);
        reducedVec = typecast(uint32(reducedNumBits), 'single');  % Convert back to float
    end 
end

function testEarlyRejectionWithReducedPrecision(embeddings, ~, ~, mode)
    % Limit dataset size to 100,000 vectors
    maxVectors = 100000;
    if size(embeddings, 1) > maxVectors
        fprintf('[Note] Reducing dataset from %d to %d vectors for performance.\n', size(embeddings, 1), maxVectors);
        selectedIndices = randperm(size(embeddings, 1), maxVectors);
        embeddings = embeddings(selectedIndices, :);
    end
    numSamples = 20;
    topK = 20;
    thetas = 0:1:1;
    numRuns = 10;
    mantissaList = 8:-1:0;
    expMantissaCombos = [5 * ones(1, numel(mantissaList)); mantissaList];
    bandwidths = expMantissaCombos(1,:) + expMantissaCombos(2,:);
    numSettings = length(bandwidths);

    if strcmp(mode, 'cosine')
        embeddings = embeddings ./ vecnorm(embeddings, 2, 2);
    end

    numVectors = size(embeddings, 1);

    queryIndices = randperm(numVectors, min(25, numVectors));
    numQueries = length(queryIndices);

    randIdxList = cell(numQueries, 1);
    randSimList = cell(numQueries, 1);
    trueTopList = cell(numQueries, 1);
    for qi = 1:numQueries
        idx = queryIndices(qi);
        [randIdx, randSim] = findRandomSimilarities(embeddings, idx, numSamples, mode);
        [trueTop, ~] = findTopKTrue(embeddings, idx, topK, mode);
        randIdxList{qi} = randIdx;
        randSimList{qi} = randSim;
        trueTopList{qi} = trueTop;
    end

    recallMatrix = zeros(length(thetas) + 1, numSettings);
    falsePosMatrixEarly = zeros(length(thetas), numSettings);
    falsePosMatrixFull = zeros(1, numSettings);

    if strcmp(mode, 'euclidean') || strcmp(mode, 'cosine')
        fprintf('\n[No Bound] Full-Precision Cutoff (No Early Rejection)\n');
        for s = 1:numSettings
            exponentBits = expMantissaCombos(1,s);
            mantissaBits = expMantissaCombos(2,s);
            totalMatches = 0;
            falsePosTotal = 0;

            reducedEmbeddings = reduceAllEmbeddings(embeddings, exponentBits, mantissaBits);

            for run = 1:numRuns
                matches = zeros(numQueries, 1);
                for qi = 1:numQueries
                    idx = queryIndices(qi);
                    [updatedIdx, ~, ~, falsePos] = updateTopKWithFullPrecisionCutoff(embeddings, reducedEmbeddings, idx, randIdxList{qi}, randSimList{qi}, topK, mode);
                    matches(qi) = sum(ismember(updatedIdx, trueTopList{qi}));
                    falsePosTotal = falsePosTotal + falsePos;
                end
                totalMatches = totalMatches + mean(matches);
            end

            recall = 100 * totalMatches / (numRuns * topK);
            recallMatrix(end, s) = recall;
            falsePosMatrixFull(s) = falsePosTotal/(numRuns*numQueries);
            fprintf('  Mantissa = %d, Exp = %d | Recall@20 = %.3f%% | False Positives = %d\n', ...
                mantissaBits, exponentBits, recall, falsePosMatrixFull(s));
        end
    end

    fprintf('\nRunning early rejection with reduced precision...\n');
    for t = 1:length(thetas)
        theta = thetas(t);
        fprintf('\n[Early Rejection] Theta = %.1f\n', theta);
        for s = 1:numSettings
            exponentBits = expMantissaCombos(1,s);
            mantissaBits = expMantissaCombos(2,s);
            totalMatches = 0;
            falsePosTotal = 0;

            reducedEmbeddings = reduceAllEmbeddings(embeddings, exponentBits, mantissaBits);

            for run = 1:numRuns
                matches = zeros(numQueries, 1);
                for qi = 1:numQueries
                    idx = queryIndices(qi);
                    [updatedIdx, ~, ~, falsePos] = updateTopKWithEarlyRejectionBitTrunc(...
                        embeddings, reducedEmbeddings, idx, ...
                        randIdxList{qi}, randSimList{qi}, topK, mode, theta);
                    matches(qi) = sum(ismember(updatedIdx, trueTopList{qi}));
                    falsePosTotal = falsePosTotal + falsePos;
                end
                totalMatches = totalMatches + mean(matches);
            end

            recall = 100 * totalMatches / (numRuns * topK);
            recallMatrix(t, s) = recall;
            falsePosMatrixEarly(t, s) = falsePosTotal / (numRuns * numQueries);
            fprintf('  Mantissa = %d, Exp = %d | Recall@20 = %.3f%% | False Positives = %d\n', ...
                mantissaBits, exponentBits, recall, falsePosMatrixEarly(t, s));
        end
    end


    %% plotting
    figure; hold on;
    markers = {'o', 's', 'd', '^', 'v', '>', '<'};
    labels = ["Theta = 0.0", "Theta = 0.2", "Theta = 0.4", ...
              "Theta = 0.6", "Theta = 0.8", "Theta = 1.0", ...
              "No Bound"];

    for i = 1:size(recallMatrix,1)
        plot(bandwidths, recallMatrix(i,:), '-o', ...
            'Marker', markers{i}, 'DisplayName', labels(i), ...
            'LineWidth', 1.8, 'MarkerSize', 6);
    end

    xlabel('Bandwidth (Remaining Exponent Bits + Mantissa Bits)');
    ylabel('Recall@20 (%)');
    title('Recall@20 vs Bandwidth Across Theta Values');
    legend('Location', 'southeast');
    grid on;
    ylim([85 100]);
    xlim([min(bandwidths) max(bandwidths)]);
    drawnow;

    figure; hold on;
    for i = 1:size(falsePosMatrixEarly,1)
        plot(bandwidths, falsePosMatrixEarly(i,:), '-o', ...
            'Marker', markers{i}, 'DisplayName', labels(i), ...
            'LineWidth', 1.8, 'MarkerSize', 6);
    end
    if strcmp(mode, 'cosine')
        plot(bandwidths, falsePosMatrixFull, '-o', ...
            'Marker', markers{end}, 'DisplayName', labels(end), ...
            'LineWidth', 1.8, 'MarkerSize', 6);
    end

    xlabel('Bandwidth (Remaining Exponent Bits + Mantissa Bits)');
    ylabel('False Positive Count');
    title('False Positives vs Bandwidth Across Theta Values');
    legend('Location', 'northwest');
    grid on;
    drawnow;
end 

function reduced = reduceAllEmbeddings(embeddings, expBits, mantBits)
    reduced = zeros(size(embeddings), 'single');
    for i = 1:size(embeddings, 1)
        reduced(i, :) = reducePrecision(embeddings(i, :), expBits, mantBits);
    end
end

function [updatedIdx, updatedSims, simCount, falsePositiveCount] = updateTopKWithFullPrecisionCutoff(fullX, reducedX, qIdx, randIdx, randSim, topK, mode)
    % Initialize the current top-K list using the random subset
    updatedIdx = randIdx;
    updatedSims = randSim;
    simCount = 0;
    falsePositiveCount = 0;
    total = 0;


    % Extract the full and reduced precision version of the query vector
    q_full = fullX(qIdx, :);

    % Candidates to consider for potential top-K inclusion (excluding query and initial sample)
    allIdx = setdiff(1:size(fullX, 1), [qIdx, randIdx]);

    for i = 1:length(allIdx)
        cIdx = allIdx(i);  % candidate index
        c_full = fullX(cIdx, :);     % full precision candidate
        c_tilde = reducedX(cIdx, :); % reduced precision candidate
        if strcmp(mode, 'cosine')
            % Compute reduced-precision cosine similarity
            sim_red = dot(q_full, c_tilde);
            [worstSim, worstIdx] = min(updatedSims);  % current worst in top-K 

            % Only compute full similarity if candidate passes reduced sim threshold
            if sim_red >= worstSim
                simCount = simCount + 1;
                sim_full = dot(q_full, c_full);  % compute full-precision sim
                if sim_full > worstSim  % replace if better than current worst
                    updatedSims(worstIdx) = sim_full;
                    updatedIdx(worstIdx) = cIdx;
                else
                    falsePositiveCount = falsePositiveCount + 1;
                end
            end

        elseif strcmp(mode, 'euclidean')
            % Compute reduced-precision Euclidean distance
            dist_red = norm(q_full - c_tilde);
            [worstDist, worstIdx] = max(updatedSims);  % current worst in top-K


            % Only compute full distance if candidate passes reduced dist threshold
            if dist_red <= worstDist
                simCount = simCount + 1;
                dist_full = norm(q_full - c_full);  % compute full-precision dist
                if dist_full < worstDist  % replace if better than current worst
                    updatedSims(worstIdx) = dist_full;
                    updatedIdx(worstIdx) = cIdx;
                    total = total + 1;
                else
                    falsePositiveCount = falsePositiveCount + 1;
                end
            end
        end
    end
end

function [updatedIdx, updatedSims, simCount, falsePositiveCount] = updateTopKWithEarlyRejectionBitTrunc(fullX, reducedX, qIdx, randIdx, randSim, topK, mode, theta)
    % Initialize top-K indices and similarities with random candidates
    updatedIdx = randIdx;
    updatedSims = randSim;
    simCount = 0;  % Count how many full-precision comparisons are made
    falsePositiveCount = 0;


    % Get full-precision and reduced-precision vectors for the query
    q_full = fullX(qIdx, :);

    % Determine candidate pool (exclude query and initial top-K)
    allIdx = setdiff(1:size(fullX, 1), [qIdx, randIdx]);

    if strcmp(mode, 'cosine')
        prevWorstIdx = -1;  % Used to track when worst vector in top-K changes

        for i = 1:length(allIdx)
            cIdx = allIdx(i);
            c_tilde = reducedX(cIdx, :);  % Reduced-precision candidate vector

            % Find current worst similarity in top-K and its index
            [minSim, minIdx] = min(updatedSims);
            wIdx = updatedIdx(minIdx);  % Index of current worst vector

            % Recompute eta_s only if worst vector has changed
            if wIdx ~= prevWorstIdx
                w_full = fullX(wIdx, :);
                w_tilde = reducedX(wIdx, :);
                eps = abs(w_full - w_tilde);  % Max deviation per dimension
                ds_q_wtilde = dot(q_full, w_tilde);  % Reduced dot product
                eta_s = ds_q_wtilde - theta*sqrt(sum((abs(q_full) .* eps).^2));  % Lower bound
                prevWorstIdx = wIdx;
            end

            % Approximate similarity check
            sim_approx = dot(q_full, c_tilde);
            if sim_approx < eta_s
                continue;  % Early rejection
            end

            % Perform full similarity check if not rejected
            full_sim = dot(q_full, fullX(cIdx, :));
            simCount = simCount + 1;

            % Replace worst if this candidate is better
            if full_sim > minSim
                updatedSims(minIdx) = full_sim;
                updatedIdx(minIdx) = cIdx;
            else
                falsePositiveCount = falsePositiveCount + 1;
            end
        end

    elseif strcmp(mode, 'euclidean')
        prevWorstIdx = -1;  % Used to track when worst vector in top-K changes

        for i = 1:length(allIdx)
            cIdx = allIdx(i);
            c_tilde = reducedX(cIdx, :);  % Reduced-precision candidate vector

            % Find current worst distance in top-K
            [worstVal, worstIdx] = max(updatedSims);
            wIdx = updatedIdx(worstIdx);  % Index of current worst vector

            % Recompute eta_e only if worst vector has changed
            if wIdx ~= prevWorstIdx
                w_full = fullX(wIdx, :);
                w_tilde = reducedX(wIdx, :);
                
                diff = q_full - w_tilde;              % Difference to approximation
                d_approx = sqrt(sum(diff.^2));                % d(q, w_tilde)
                residual = sqrt(sum((w_full - w_tilde).^2));    % ||w - w_tilde||
                
                % eta_e = d_approx + theta*residual;          % Upper bound on d_E(q, w)
                
                % eta_e = sqrt(sum(diff.^2) + theta * (sum((w_full - w_tilde).^2) + 2 * sqrt(sum(diff.^2 .* (w_full - w_tilde).^2))));
                % eta_e = sqrt(sum(diff.^2) + theta * (2 * sqrt(sum(diff.^2 .* (w_full - w_tilde).^2))));
                eta_e = sqrt(sum(diff.^2) + theta * (sqrt(sum((diff.^2) .* (w_full - w_tilde).^2))));
                
                prevWorstIdx = wIdx;
            end

            % Approximate distance check
            approx_dist = norm(q_full - c_tilde);
            if approx_dist > eta_e
                continue;  % Early rejection
            end

            % Perform full-precision distance check if not rejected
            full_dist = norm(q_full - fullX(cIdx, :));
            simCount = simCount + 1;

            % Replace worst if this candidate is better
            if full_dist < worstVal
                updatedSims(worstIdx) = full_dist;
                updatedIdx(worstIdx) = cIdx;
            else
                falsePositiveCount = falsePositiveCount + 1;
            end
        end
    end
end

function [randomIndices, randomSimilarities] = findRandomSimilarities(X, index, num, mode)
    % This function selects a random subset of `num` vectors (excluding the query vector)
    % and computes their similarity/distance to the query vector using the specified mode.

    all = setdiff(1:size(X, 1), index);

    % Randomly select `num` indices from remaining vectors
    randomIndices = all(randperm(length(all), num));

    % Extract the query vector
    q = X(index, :);

    % Preallocate similarity/distance result array
    randomSimilarities = zeros(num, 1);

    % Loop through each randomly selected index
    for i = 1:num
        c = X(randomIndices(i), :);  % candidate vector

        % Compute similarity or distance depending on mode
        if strcmp(mode, 'cosine')
            randomSimilarities(i) = dot(q, c);  % cosine similarity
        elseif strcmp(mode, 'euclidean')
            randomSimilarities(i) = norm(q - c);  % full-precision distance
        end
    end
end

function [topIdx, topVals] = findTopKTrue(X, index, k, mode)
    % This function computes the true top-K nearest neighbors (based on full-precision)
    % for the given query index using cosine similarity or euclidean distance.

    q = X(index, :);

    % Initialize an array to hold all similarity or distance scores
    scores = zeros(size(X, 1), 1);

    % Loop through every vector in X
    for i = 1:size(X, 1)
        if i == index
            % Skip self-comparison by assigning the worst possible score
            scores(i) = (strcmp(mode, 'cosine') * -Inf) + (strcmp(mode, 'euclidean') * Inf);
        else
            % Compute similarity/distance depending on mode
            if strcmp(mode, 'cosine')
                scores(i) = dot(q, X(i, :));  % cosine similarity
            else
                scores(i) = norm(q - X(i, :));  % euclidean distance
            end
        end
    end

    % Extract the top-K indices and values
    if strcmp(mode, 'cosine')
        [topVals, topIdx] = maxk(scores, k);  % top K highest similarities
    elseif strcmp(mode, 'euclidean')
        [topVals, topIdx] = mink(scores, k);  % top K smallest distances
    end
end

% ----- Dataset 1 -----
load('finewebFP_N16_M10.mat');
mBits = 4;
exBits = 5;
fprintf('\n--- Running on fineweb---\n');
testEarlyRejectionWithReducedPrecision(embeddings, mBits, exBits, 'cosine');

% ----- Dataset 2 -----
load('msmarcoFP_N16_M10.mat');
mBits = 4;
exBits = 5;
fprintf('\n--- Running on msmarco ---\n');
testEarlyRejectionWithReducedPrecision(embeddings, mBits, exBits, 'cosine');

% ----- Dataset 3 -----
load('dbpediaFP_N16_M10.mat');
mBits = 4;
exBits = 5;
fprintf('\n--- Running on dbpedia ---\n');
testEarlyRejectionWithReducedPrecision(embeddings, mBits, exBits, 'cosine');

% ----- Dataset 4 -----
load('sift_base.mat');
mBits = 4;
exBits = 5;
fprintf('\n--- Running on  sift ---\n');
testEarlyRejectionWithReducedPrecision(embeddings, mBits, exBits, 'euclidean');

% ----- Dataset 5 -----
load('gist960FP_N16_M10.mat');
mBits = 4;
exBits = 5;
fprintf('\n--- Running on  ---\n');
testEarlyRejectionWithReducedPrecision(embeddings, mBits, exBits, 'euclidean');

% ----- Dataset 6 -----
load('gloveFP_N16_M10.mat');
mBits = 4;
exBits = 5;
fprintf('\n--- Running on  ---\n');
testEarlyRejectionWithReducedPrecision(embeddings, mBits, exBits, 'euclidean');