function [inputSgl, tasksSgl, trainSgl, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask, multiCap, multiCap_con, multiCap_inc, relevantTasks, NPathways] = generateEnvironmentToGraph(A, NFeatures, samplesPerTask, sdScale, sameStimuliAcrossTasks)

NPathways = max(size(A));
samples = samplesPerTask;
     
% file name
checkFile = 0;
filename = ['trainingSets/Giovanni_Simulation_' num2str(NPathways) 'P' num2str(NFeatures) 'F.mat'];

% check if task environment already exists
if(~exist(filename, 'file') || checkFile == 0)
    
     % extract relevant tasks
     taskM = transpose(reshape(1:(NPathways^2), NPathways, NPathways));
     mask = [A zeros(size(A,1), NPathways-size(A,2)); zeros(NPathways-size(A,1), NPathways)];
     relevantTasks = taskM(mask == 1);
     
     % If the user has requested multiCap patterns as well, then calculate
     % them
     if(nargout > 8)
        [inputSgl, tasksSgl, trainSgl, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask, multiCap] = createTaskPatterns(NPathways, NFeatures, samples, sdScale, sameStimuliAcrossTasks, relevantTasks);
    
        multiCap_con = cell(1, NPathways);
        multiCap_inc = cell(1, NPathways);
        
        multiCap_con{1} = multiCap{1};
        multiCap_inc{1} = multiCap{1};

        for cap = 2:length(multiCap)

        % split multitasking conditions into congruent & incongruent conditions

%             congruency = nan(1, size(multiCap{cap}.train,1));
%             for row = 1:length(congruency)
%                 congrSum = sum(reshape(multiCap{cap}.train(row,:), NFeatures, NPathways),2);
%                 numCorrectOutputUnits = sum(multiCap{cap}.train(row,:));
%                 if(max(congrSum) == numCorrectOutputUnits)
%                     congruency(row) = 1;
%                 else
%                     congruency(row) = 0;
%                 end
%             end
%             assert(isequal(multiCap{cap}.congruency, congruency'));
            congruency = multiCap{cap}.congruency;
            
            multiCap_con{cap}.taskCombs = multiCap{cap}.taskCombs;
            multiCap_con{cap}.taskIdx = multiCap{cap}.taskIdx(congruency);
            multiCap_con{cap}.input = multiCap{cap}.input(congruency,:);
            multiCap_con{cap}.tasks = multiCap{cap}.tasks(congruency,:);
            multiCap_con{cap}.train = multiCap{cap}.train(congruency,:);
            
            multiCap_inc{cap}.taskCombs = multiCap{cap}.taskCombs;
            multiCap_inc{cap}.taskIdx = multiCap{cap}.taskIdx(~congruency);
            multiCap_inc{cap}.input = multiCap{cap}.input(~congruency,:);
            multiCap_inc{cap}.tasks = multiCap{cap}.tasks(~congruency,:);
            multiCap_inc{cap}.train = multiCap{cap}.train(~congruency,:);

            % disp(['cap ' num2str(cap) '/' num2str(length(multiCap))]);
        end
     
%        save(filename, 'inputSgl', 'tasksSgl', 'trainSgl', 'tasksIdxSgl', 'stimIdxSgl', 'inputSgl_mask', 'tasksSgl_mask', 'trainSgl_mask', 'multiCap', 'multiCap_con', 'multiCap_inc', 'relevantTasks', 'NPathways')
     else
        [inputSgl, tasksSgl, trainSgl, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask] = createTaskPatterns(NPathways, NFeatures, samples, sdScale, sameStimuliAcrossTasks, relevantTasks);
%        save(filename, 'inputSgl', 'tasksSgl', 'trainSgl', 'tasksIdxSgl', 'stimIdxSgl', 'inputSgl_mask', 'tasksSgl_mask', 'trainSgl_mask', 'relevantTasks', 'NPathways')
     end
 
else
    % load task environment
    load(filename);
end

end