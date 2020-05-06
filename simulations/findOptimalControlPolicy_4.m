function [taskCardinality, maximumPerformanceCurve, minimumPerformanceCurve, effectivePPC_restrictive, effectivePPC_unthresholded, timing, effectivePPC_permissive] = findOptimalControlPolicy_4(taskNet, NPathways, multiCap, goodPerformanceThresholds)

    optimalPerformance_meanAccuracy = [];   % stores optimal performance (using mean accuracy as performance criterion)
    optimalTaskVector_meanAccuracy = [];     % stores optimal contrl policy (using mean accuracy as performance criterion)
    taskCardinality_meanAccuracy = [];           % stores optimal number of parallel tasks (using mean accuracy as performance criterion)

    optimalPerformance_responseProbability = [];   % stores optimal performance (using response probability as performance criterion)
    optimalTaskVector_responseProbability = [];     % stores optimal contrl policy (using response probability as performance criterion)
    taskCardinality_responseProbability = [];           % stores optimal number of parallel tasks (using response probability as performance criterion)

    for cap = 1:length(multiCap)
        
        % find all available tasks
        allTaskCombs = multiCap{cap}.taskCombs;
        
        if(~isempty(allTaskCombs))
            
            taskAccuracies{cap}.absoluteAccuracy = zeros(size(allTaskCombs, 1), cap);
            taskAccuracies{cap}.PCorrect = zeros(size(allTaskCombs, 1), cap);
            taskAccuracies{cap}.LCA = zeros(size(allTaskCombs, 1), cap);

            for combIdx = 1:size(allTaskCombs, 1);

                % find corresponding test patterns
                patternIdx = multiCap{cap}.taskIdx == combIdx;
                input = multiCap{cap}.input(patternIdx, :);
                tasks = multiCap{cap}.tasks(patternIdx, :);
                train = multiCap{cap}.train(patternIdx, :);

                % Get the encoding for the task combination
                currentTaskComb = tasks(1,:);

                % check task cardinality
                if(cap ~= sum(currentTaskComb))
                    warning(['Number of tasks does not match assigned cardinality. Cardinality is ' num2str(cap) ' and number of tasks is ' num2str(sum(currenTasks))]);
                end

                % compute output for all task patterns of a task
                [outputPatterns] = taskNet.runSet(input, tasks, train);

                %% COMPUTE OPTIMIZATION CRITERION

                % identify relevant output dimension
                taskM = reshape(currentTaskComb, NPathways, NPathways);
                [relevantOutputDims relevantInputDim] = find(taskM == 1);

                % test if two tasks afford a response at the same output dimension
                if(length(unique(relevantOutputDims)) ~= length(relevantOutputDims))
                    warning('Tested multitasking pair affords response at the same output dimension.');
                end

                %% mean task accuracy as performance criterion

                tic;

                % number of features per output dimension
                NFeatures = size(train,2)/NPathways;

                % generate mask to extract relevant output dimension
                relevantOutputMask = zeros(NFeatures, NPathways);
                relevantOutputMask(:, relevantOutputDims) = repmat(relevantOutputDims', NFeatures, 1);
                relevantOutputMask = relevantOutputMask(:)';

                summedTaskAccuracy_meanAccuracy = 0;

                meanTaskError = mean(abs(outputPatterns -  train), 1);

                for i = 1:length(relevantOutputDims)

                    dim = relevantOutputDims(i);
                    taskAccuracies{cap}.absoluteAccuracy(combIdx, i) = 1 - mean(meanTaskError(relevantOutputMask == dim));

                end

                timing.findOptimalControlPolicy_absoluteAccuracy = toc;


                %% response probability as performance criterion

                tic;

                summedTaskAccuracy_responseProbability = 0;

                outputDim = mod(allTaskCombs(combIdx,:)-1,NPathways)+1;
                PCorrect_tasks = zeros(size(outputPatterns,1), length(outputDim));
                for ii=1:length(outputDim)
                   relevantFeatures = (NFeatures*(outputDim(ii)-1)+1) : (NFeatures*outputDim(ii));

                   % Get the correct feature dimension respones
                   [~, correctResponse] = max(train(:,relevantFeatures), [], 2);

                   % Convert pathway feature dimension index to matrix column
                   % of train
                   correctResponse = relevantFeatures(correctResponse);

                   % Compute the softmaxed value of just the correct response
                   % dimension. We need to convert the row and column indicies
                   % we have to a linear index first.
                   idx = (1:size(outputPatterns,1)) + (correctResponse-1)*size(outputPatterns,1);
                   outcomeProb = outputPatterns(idx) ./ sum(outputPatterns(:,relevantFeatures), 2)';

                   PCorrect_tasks(:, ii) = outcomeProb;
                end

                taskAccuracies{cap}.PCorrect(combIdx, :) = mean(PCorrect_tasks);

                timing.findOptimalControlPolicy_PCorrect = toc;

                %% optimal LCA accuracy as performance criterion

                tic; 

                % COMPUTE LCA ACCURACY

                % LCA settings
                LCA_settings.dt_tau = 0.1;
                LCA_settings.maxTimeSteps = 100; %50;        
                LCA_settings.numSimulations = 1000;   
                LCA_settings.lambda = 0.4;                      % leakage
                LCA_settings.alpha = 0.2;                         % recurrent excitation
                LCA_settings.beta = 0.2;                           % inhibition
                LCA_settings.responseThreshold = [0:0.1:0.5];         % tested threshold
                LCA_settings.T0 = 0.15;                                         % non-decision time
                LCA_settings.W_ext = eye(NFeatures) * 0.5;           % weight of external input
                LCA_settings.c = 0.1;                                       % noise

                % LCA call
                [optTaskAccuracy, ~, ~, ~, optThreshIdx] ...
                = taskNet.runLCA_RDist(LCA_settings, input, tasks, train);

                disp(['max thresh: ' num2str(max(max(optThreshIdx)))]);

                taskIDs = find(currentTaskComb == 1);

                taskAccuracies{cap}.LCA(combIdx, :) = nanmean(optTaskAccuracy(:, taskIDs),1);

                timing.findOptimalControlPolicy_LCA = toc;

                disp(['cap: ' num2str(cap) ', comb ' num2str(combIdx) '/' num2str(size(allTaskCombs, 1))]);
            end

        end
    end
    
    %% determine parallel processing capacity (task cardinality) and effective parallel processing capacity
    
    numPerformanceThresholds = length(goodPerformanceThresholds);
    
    taskCardinality.absoluteAccuracy = zeros(1, numPerformanceThresholds);
    taskCardinality.PCorrect = zeros(1, numPerformanceThresholds);
    taskCardinality.LCA = zeros(1, numPerformanceThresholds);
    
    effectivePPC_restrictive.absoluteAccuracy = zeros(length(taskAccuracies), numPerformanceThresholds);
    effectivePPC_restrictive.PCorrect = zeros(length(taskAccuracies), numPerformanceThresholds);
    effectivePPC_restrictive.LCA = zeros(length(taskAccuracies), numPerformanceThresholds);
    
    effectivePPC_permissive.absoluteAccuracy = zeros(length(taskAccuracies), numPerformanceThresholds);
    effectivePPC_permissive.PCorrect = zeros(length(taskAccuracies), numPerformanceThresholds);
    effectivePPC_permissive.LCA = zeros(length(taskAccuracies), numPerformanceThresholds);
    
    effectivePPC_unthresholded.absoluteAccuracy = zeros(length(taskAccuracies),1);
    effectivePPC_unthresholded.PCorrect = zeros(length(taskAccuracies), 1);
    effectivePPC_unthresholded.LCA = zeros(length(taskAccuracies), 1);
    
    
    % for each performance threshold
    for threshIdx = 1:length(goodPerformanceThresholds)
        
        threshold = goodPerformanceThresholds(threshIdx);
        
        % for all capacities, check if there is at least one task combination that meets performance threshold
        for cap = 1:length(taskAccuracies)

            goodPerformance = any(all(taskAccuracies{cap}.absoluteAccuracy > threshold, 2));
            if(goodPerformance)
                taskCardinality.absoluteAccuracy(threshIdx) = cap;
            end
            effectivePPC_restrictive.absoluteAccuracy(cap, threshIdx) = mean(all(taskAccuracies{cap}.absoluteAccuracy > threshold, 2));
            effectivePPC_permissive.absoluteAccuracy(cap, threshIdx) = mean(mean(taskAccuracies{cap}.absoluteAccuracy > threshold, 2));
            effectivePPC_unthresholded.absoluteAccuracy(cap, 1) = mean(mean(taskAccuracies{cap}.absoluteAccuracy));
            
            goodPerformance = any(all(taskAccuracies{cap}.PCorrect > threshold, 2));
            if(goodPerformance)
                taskCardinality.PCorrect(threshIdx) = cap;
            end
            effectivePPC_restrictive.PCorrect(cap, threshIdx) = mean(all(taskAccuracies{cap}.PCorrect > threshold, 2));
            effectivePPC_permissive.PCorrect(cap, threshIdx) = mean(mean(taskAccuracies{cap}.PCorrect > threshold, 2));
            effectivePPC_unthresholded.PCorrect(cap, 1) = mean(mean(taskAccuracies{cap}.PCorrect));
            
            goodPerformance = any(all(taskAccuracies{cap}.LCA > threshold, 2));
            if(goodPerformance)
                taskCardinality.LCA(threshIdx) = cap;
            end
            effectivePPC_restrictive.LCA(cap, threshIdx) = mean(all(taskAccuracies{cap}.LCA > threshold, 2));
            effectivePPC_permissive.LCA(cap, threshIdx) = mean(mean(taskAccuracies{cap}.LCA > threshold, 2));
            effectivePPC_unthresholded.LCA(cap, threshIdx) = mean(mean(taskAccuracies{cap}.LCA));
            

        end
        
    end

    %% determine maximum and minimum performance curve
    
    maximumPerformanceCurve.absoluteAccuracy = nan(1, length(multiCap));
    maximumPerformanceCurve.PCorrect = nan(1, length(multiCap));
    maximumPerformanceCurve.LCA = nan(1, length(multiCap));
    
    minimumPerformanceCurve.absoluteAccuracy = nan(1, length(multiCap));
    minimumPerformanceCurve.PCorrect = nan(1, length(multiCap));
    minimumPerformanceCurve.LCA = nan(1, length(multiCap));
    
    % for all capacities, look for minimum and maximum performance of the worst task in a set 
    for cap = 1:length(taskAccuracies)
        
            maximumPerformanceCurve.absoluteAccuracy(cap) = max(min(taskAccuracies{cap}.absoluteAccuracy, [], 2));
            maximumPerformanceCurve.PCorrect(cap) = max(min(taskAccuracies{cap}.PCorrect, [], 2));
            maximumPerformanceCurve.LCA(cap) = max(min(taskAccuracies{cap}.LCA, [], 2));
        
            minimumPerformanceCurve.absoluteAccuracy(cap) = min(min(taskAccuracies{cap}.absoluteAccuracy, [], 2));
            minimumPerformanceCurve.PCorrect(cap) = min(min(taskAccuracies{cap}.PCorrect, [], 2));
            minimumPerformanceCurve.LCA(cap) = min(min(taskAccuracies{cap}.LCA, [], 2));
    end
    
    
    
end