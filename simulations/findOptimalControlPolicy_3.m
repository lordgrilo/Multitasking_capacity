function [optimalTaskVector_meanAccuracy taskCardinality_meanAccuracy optimalPerformance_meanAccuracy optimalTaskVector_responseProbability taskCardinality_responseProbability optimalPerformance_responseProbability] = findOptimalControlPolicy_3(taskNet, NPathways, multiCap, goodPerformanceThresh)

    optimalPerformance_meanAccuracy = [];   % stores optimal performance (using mean accuracy as performance criterion)
    optimalTaskVector_meanAccuracy = [];     % stores optimal contrl policy (using mean accuracy as performance criterion)
    taskCardinality_meanAccuracy = [];           % stores optimal number of parallel tasks (using mean accuracy as performance criterion)

    optimalPerformance_responseProbability = [];   % stores optimal performance (using response probability as performance criterion)
    optimalTaskVector_responseProbability = [];     % stores optimal contrl policy (using response probability as performance criterion)
    taskCardinality_responseProbability = [];           % stores optimal number of parallel tasks (using response probability as performance criterion)

    for cap = 1:length(multiCap)
        
        % find all available tasks
        allTaskCombs = multiCap{cap}.taskCombs;
        
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
                taskAccuracy = 1 - mean(meanTaskError(relevantOutputMask == dim));
                if(taskAccuracy < goodPerformanceThresh)
                    taskAccuracy = - 1;
                end
                summedTaskAccuracy_meanAccuracy = summedTaskAccuracy_meanAccuracy + taskAccuracy;

            end
            
            % define optimization criterion
            % If capacity is higher than previous capcity and performance
            % criterion is met, then select new control policy. If no optimal 
            % performance assigned, then use current control policy as starting point
            if(isempty(optimalPerformance_meanAccuracy) || (summedTaskAccuracy_meanAccuracy > optimalPerformance_meanAccuracy) )

                optimalPerformance_meanAccuracy = summedTaskAccuracy_meanAccuracy;
                optimalTaskVector_meanAccuracy = currentTaskComb;
                taskCardinality_meanAccuracy = cap;

            end
            
            %% response probability as performance criterion
            
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
            
            PCorrect_tasks_mean = mean(PCorrect_tasks,1);
                  
            for i = 1:length(allTaskCombs(combIdx,:));
                
                taskAccuracy = PCorrect_tasks_mean(i);
                
                if(taskAccuracy < goodPerformanceThresh)
                    taskAccuracy = - 1;
                end
                summedTaskAccuracy_responseProbability = summedTaskAccuracy_responseProbability + taskAccuracy;

            end

            % define optimization criterion
            % If capacity is higher than previous capcity and performance
            % criterion is met, then select new control policy. If no optimal 
            % performance assigned, then use current control policy as starting point
            if(isempty(optimalPerformance_responseProbability) || (summedTaskAccuracy_responseProbability > optimalPerformance_responseProbability) )

                optimalPerformance_responseProbability = summedTaskAccuracy_responseProbability;
                optimalTaskVector_responseProbability = currentTaskComb;
                taskCardinality_responseProbability = cap;

            end

            %%
        
        end
        
    end

end