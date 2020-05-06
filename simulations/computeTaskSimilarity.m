function [hiddenSimilarity, outputSimilarity] = computeTaskSimilarity(taskNet, inputPatterns, taskPatterns, varargin)
% [similarity] = computeTaskSimilarity(taskNet, inputPatterns, taskPatterns, optionalParameters)
%
% computes the similarity of the hidden and output representations between
% tasks using the NNmodel taskNet for the given inputPatterns and
% taskPatterns.
%
% It is possible to specify which tasks shall be computed in
% optionalParameters. E.g. 
% computeTaskSimilarity(taskNet, inputPatterns, taskPatterns, 'tasks', [1 2]) 
% computes the simiarity between tasks 1 & 2.
%
% One may also specify the similarity metric in optionalParameters, e.g.
% computeTaskSimilarity(taskNet, inputPatterns, taskPatterns, 'similarityMetric', 'CorrTaskMean') 
% 
% The following similarity metrics are permitted:
%
% 'CorrTaskAvg' ... correlation of mean hidden unit activity for a given tasks (averaged
% across all stimuli)
%
%
% 'AvgTaskCorr' ... mean correlation for the activations of different tasks under the
% same stimulus
%
% Finally it is possible to specify the parameters of the similarity
% function, e.g. computeTaskSimilarity(taskNet, inputPatterns, taskPatterns, 'similarityMetric', 'CorrTaskMean', 'param', 'Pearson') 
%
% Author: Sebastian Musslick

    % similarityMeasures
    similarityMetricsIdentifiers = {'CorrTaskAvg', 'AvgTaskCorr'};
    
    % default similarity metric
    default_similarityMetric = 'CorrTaskAvg';
    default_params{1} = 'Mean';
    default_params{2} = 'Pearson';
    
    % by default compute similarities for all tasks
    default_relevantTasks_mask = 1:size(taskPatterns, 2);
    default_relevantTasks = default_relevantTasks_mask(logical(sum(unique(taskPatterns, 'rows'))));

    %% parse input arguments
    similarityMetric = nan;
    params = {};
    relevantTasks = [];
    
    argumentIdx = 1;
    while argumentIdx <= length(varargin)
        
        if(ischar(varargin{argumentIdx}))
            
            stringVar = varargin{argumentIdx};
            
            if(strcmp(stringVar, 'tasks'))  % identified tasks
                if((argumentIdx + 1) <= length(varargin))
                    argumentIdx = argumentIdx + 1;
                    if(isnumeric(varargin{argumentIdx}))
                        relevantTasks = varargin{argumentIdx};
                    end
                end
            elseif(strcmp(stringVar, 'similarityMetric')) % identified similarity metric
                
                if((argumentIdx + 1) <= length(varargin))
                    argumentIdx = argumentIdx + 1;
                    stringVar = varargin{argumentIdx};
                    if(ismember(stringVar, similarityMetricsIdentifiers))
                        similarityMetric = stringVar;
                    end
                end
                
            elseif(strcmp(stringVar, 'param')) % identified similarity metric parameter
                while((argumentIdx + 1) <= length(varargin))
                    argumentIdx = argumentIdx + 1;
                    params{length(params)+1} = varargin{argumentIdx};
                end
            end
            
        end
        
        argumentIdx = argumentIdx + 1;
        
    end
    
    % handle empty inputs
    if(isempty(relevantTasks))
        relevantTasks = default_relevantTasks;
    end
    
    if(isnan(similarityMetric))
        similarityMetric = default_similarityMetric;
        params = default_params;
    end
        
    %% compute similarities
    
    switch similarityMetric
        case 'CorrTaskAvg'
            [hiddenSimilarity, outputSimilarity] = CorrTaskAvg(taskNet, inputPatterns, taskPatterns, relevantTasks, params{1}, params{2});
        case 'AvgTaskCorr'
            [hiddenSimilarity, outputSimilarity] = MeanTaskCorr(taskNet, inputPatterns, taskPatterns, relevantTasks, params{1}, params{2});
    end



end

% 'CorrTaskAvg' ... mean hidden unit activity for a given tasks (averaged
% across all stimuli)
function [hiddenSimilarity, outputSimilarity] = CorrTaskAvg(taskNet, inputPatterns, taskPatterns, relevantTasks, avgType, corrType)

           nLayers = length(taskNet.Nhidden);
           [outputData, hiddenData] = taskNet.runSet(inputPatterns, taskPatterns);
           
           % determine task IDs
           taskIDs = nan(1, size(taskPatterns, 1));
           for i = 1:size(taskPatterns,1)
               taskIDs(i) = find(taskPatterns(i, :) == 1);
           end
           
           % compute average task activity at hidden layer
           if(nLayers > 1)
               for layer = 1:nLayers
                    switch avgType
                       case 'Mean'
                            GroupMean=arrayfun(@(k) transpose(mean(hiddenData{layer}(taskIDs == relevantTasks(k),:),1)) ,1:length(relevantTasks), 'UniformOutput', 0);
                       case 'Median'
                            GroupMean=arrayfun(@(k) transpose(median(hiddenData{layer}(taskIDs == relevantTasks(k),:),1)) ,1:length(relevantTasks), 'UniformOutput', 0);
                   end
                   hiddenTaskAvg{layer}=[GroupMean{1:length(relevantTasks)}];
               end
           else
               switch avgType
                   case 'Mean'
                        GroupMean=arrayfun(@(k) transpose(mean(hiddenData(taskIDs == relevantTasks(k),:),1)) ,1:length(relevantTasks), 'UniformOutput', 0);
                   case 'Median'
                        GroupMean=arrayfun(@(k) transpose(median(hiddenData(taskIDs == relevantTasks(k),:),1)) ,1:length(relevantTasks), 'UniformOutput', 0);
               end
               hiddenTaskAvg=[GroupMean{1:length(relevantTasks)}];
           end
           
           % compute average task activity at output layer
           switch avgType
               case 'Mean'
                    GroupMean=arrayfun(@(k) transpose(mean(outputData(taskIDs == relevantTasks(k),:),1)) ,1:length(relevantTasks), 'UniformOutput', 0);
               case 'Median'
                   GroupMean=arrayfun(@(k) transpose(median(outputData(taskIDs == relevantTasks(k),:),1)) ,1:length(relevantTasks), 'UniformOutput', 0);
           end
           outputTaskAvg=[GroupMean{1:length(relevantTasks)}];
           
           % center data 
           if(nLayers > 1)
               for layer = 1:nLayers
                   switch avgType
                    case 'Mean'
                        hiddenTaskAvg{layer} = hiddenTaskAvg{layer} - repmat(mean(hiddenTaskAvg{layer},2) ,1, size(hiddenTaskAvg{layer},2));
                        outputTaskAvg = outputTaskAvg - repmat(mean(outputTaskAvg,2) ,1, size(outputTaskAvg,2));
                   case 'Median'
                       hiddenTaskAvg{layer} = hiddenTaskAvg{layer} - repmat(median(hiddenTaskAvg{layer},2) ,1, size(hiddenTaskAvg{layer},2));
                       outputTaskAvg = outputTaskAvg - repmat(median(outputTaskAvg,2) ,1, size(outputTaskAvg,2));
               end
               end
           else
               switch avgType
                    case 'Mean'
                        hiddenTaskAvg = hiddenTaskAvg - repmat(mean(hiddenTaskAvg,2) ,1, size(hiddenTaskAvg,2));
                        outputTaskAvg = outputTaskAvg - repmat(mean(outputTaskAvg,2) ,1, size(outputTaskAvg,2));
                   case 'Median'
                       hiddenTaskAvg = hiddenTaskAvg - repmat(median(hiddenTaskAvg,2) ,1, size(hiddenTaskAvg,2));
                       outputTaskAvg = outputTaskAvg - repmat(median(outputTaskAvg,2) ,1, size(outputTaskAvg,2));
               end
           end

           
           % compute similarity between average task activity vectors
           if(nLayers > 1)
               for layer = 1:nLayers
                    switch corrType
                       case 'Pearson'
                            hiddenSimilarity{layer} = corr(hiddenTaskAvg{layer}, 'type', corrType);
                            outputSimilarity = corr(outputTaskAvg, 'type', corrType);
                       case 'Spearman'
                           [hiddenSimilarity{layer}, p_hidden] = corr(hiddenTaskAvg{layer}, 'type', corrType, 'tail', 'right');
                           hiddenSimilarity{layer}(p_hidden > 0.001) = 0;       % cut off insignificant correlations
                           hiddenSimilarity{layer}(p_hidden < 0.001) = 1;       % set significant correlations to 1
                           hiddenSimilarity{layer}(eye(size(hiddenSimilarity)) == 1) =  1;
                           [outputSimilarity, p_output] = corr(outputTaskAvg, 'type', corrType, 'tail', 'right');
                           outputSimilarity(p_output > 0.001) = 0;        % cut off insignificant correlations
                           outputSimilarity(p_output < 0.001) = 1;        % set significant correlations to 1
                           outputSimilarity(eye(size(outputSimilarity)) == 1) =  1;        
                   end
               end
           else
               switch corrType
                   case 'Pearson'
                        hiddenSimilarity = corr(hiddenTaskAvg, 'type', corrType);
                        outputSimilarity = corr(outputTaskAvg, 'type', corrType);
                   case 'Spearman'
                       [hiddenSimilarity, p_hidden] = corr(hiddenTaskAvg, 'type', corrType, 'tail', 'right');
                       hiddenSimilarity(p_hidden > 0.001) = 0;       % cut off insignificant correlations
                       hiddenSimilarity(p_hidden < 0.001) = 1;       % set significant correlations to 1
                       hiddenSimilarity(eye(size(hiddenSimilarity)) == 1) =  1;
                       [outputSimilarity, p_output] = corr(outputTaskAvg, 'type', corrType, 'tail', 'right');
                       outputSimilarity(p_output > 0.001) = 0;        % cut off insignificant correlations
                       outputSimilarity(p_output < 0.001) = 1;        % set significant correlations to 1
                       outputSimilarity(eye(size(outputSimilarity)) == 1) =  1;        
               end
           end

end

% 'MeanTaskCorr' ... mean correlation for the activations of different tasks under the same stimulus
function [hiddenSimilarity, outputSimilarity] = MeanTaskCorr(taskNet, inputPatterns, taskPatterns, relevantTasks, avgType, corrType)

        nLayers = length(taskNet.Nhidden);
        
        % remove duplicate patterns
        uniquePatterns = unique([inputPatterns taskPatterns], 'rows');
        inputPatterns = uniquePatterns(:, 1:size(inputPatterns,2));
        taskPatterns = uniquePatterns(:, (size(inputPatterns,2)+1):end);

        [outputData hiddenData] = taskNet.runSet(inputPatterns, taskPatterns);
        
        % determine all unique stimuli
        stimuli = unique(inputPatterns, 'rows');
        
        % determine all tasks
        relevantTaskPatterns = zeros(length(relevantTasks), size(taskPatterns,2));
        for relTaskIdx = 1:length(relevantTasks)
            relevantTaskPatterns(relTaskIdx, relevantTasks(relTaskIdx)) = 1;
        end
        
        % for each stimulus, compute correlation between tasks
        if(nLayers > 1)
               for layer = 1:nLayers
                    stimHiddenR{layer} = nan(length(relevantTasks), length(relevantTasks), size(stimuli, 1));
                    stimOutputR = nan(length(relevantTasks), length(relevantTasks), size(stimuli, 1));
               end
        else
            stimHiddenR = nan(length(relevantTasks), length(relevantTasks), size(stimuli, 1));
            stimOutputR = nan(length(relevantTasks), length(relevantTasks), size(stimuli, 1));
        end
        for stimIdx = 1:size(stimuli, 1)
            
            currentStimulus = stimuli(stimIdx, :);
            
            stimPatternIdx = ismember(inputPatterns, currentStimulus, 'rows');
            taskPatternIdx = ismember(taskPatterns, relevantTaskPatterns, 'rows');
            
            selectedPatterns = stimPatternIdx & taskPatternIdx;
            selectedTaskPatterns = taskPatterns(selectedPatterns, :);
            
            % make sure all patterns are unique both with respect to
            % available stimuli and with respect to available tasks
            if(size(relevantTaskPatterns,1) ~= size(selectedTaskPatterns, 1))
                error('For a given task there is more than one stimulus of the same kind available. Not specified what to do in this case.');
            end
            
            % compute correlation matrix
            if(nLayers > 1)
               for layer = 1:nLayers
                     switch corrType
                       case 'Pearson'
                            stimHiddenR{layer}(:,:,stimIdx) = corr(transpose(hiddenData{layer}(selectedPatterns,:)), 'type', corrType);
                            stimOutputR(:,:,stimIdx) = corr(transpose(outputData(selectedPatterns,:)), 'type', corrType);
                      case 'Spearman'
                            stimHiddenR{layer}(:,:,stimIdx) = corr(transpose(hiddenData{layer}(selectedPatterns,:)), 'type', corrType, 'tail', 'right');
                            stimOutputR(:,:,stimIdx) = corr(transpose(outputData(selectedPatterns,:)), 'type', corrType, 'tail', 'right');
                    end
               end
            else
                switch corrType
                   case 'Pearson'
                        stimHiddenR(:,:,stimIdx) = corr(transpose(hiddenData(selectedPatterns,:)), 'type', corrType);
                        stimOutputR(:,:,stimIdx) = corr(transpose(outputData(selectedPatterns,:)), 'type', corrType);
                  case 'Spearman'
                        stimHiddenR(:,:,stimIdx) = corr(transpose(hiddenData(selectedPatterns,:)), 'type', corrType, 'tail', 'right');
                        stimOutputR(:,:,stimIdx) = corr(transpose(outputData(selectedPatterns,:)), 'type', corrType, 'tail', 'right');
                end
            end
            
        end
        
        if(nLayers > 1)
               for layer = 1:nLayers
                    switch avgType
                        case 'Mean'
                            hiddenSimilarity{layer}  = mean(stimHiddenR{layer}, 3);
                            outputSimilarity  = mean(stimOutputR, 3);
                        case 'Median'
                            hiddenSimilarity{layer}  = median(stimHiddenR{layer}, 3);
                            outputSimilarity  = median(stimOutputR, 3);
                   end
               end
        else
            switch avgType
                        case 'Mean'
                            hiddenSimilarity  = mean(stimHiddenR, 3);
                            outputSimilarity  = mean(stimOutputR, 3);
                        case 'Median'
                            hiddenSimilarity  = median(stimHiddenR, 3);
                            outputSimilarity  = median(stimOutputR, 3);
            end
        end
        
end


        