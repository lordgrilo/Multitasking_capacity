function [inputSgl, tasksSgl, trainSgl, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask, multiCap] = createTaskPatterns_GenG(NPathways, NFeatures, samples, sdScale, sameStimuliAcrossTasks, varargin)
% createTrainingPatterns generate single tasking and multitasking training
% patterns
% 
% INPUT ARGUMENTS:
%
% NPathways                        ...Number of pathways (i.e. number of feature dimensions & output dimensions)
% NFeatures                         ...Number of features per dimension
% samples                           ...Number of stimulus samples per task.
%                                           if samples = [], then all
%                                           possible samples wil be
%                                           generated for this task
%                                           environment
% sdScale                            ...the maximum standard deviation that is used to sample
%                                           each stimulus feature
% sameStimuliAcrossTasks    ...If sameStimuliAcrossTasks is set to 1, then
%                                           each task is paired with the
%                                           same input stimuli (in inputSgl). If
%                                           sameStimuliAcrossTasks is set
%                                           to 0 then input stimuli (in inputSgl) can
%                                           differ across tasks.
%
% OPTIONAL INPUT ARGUMENTS: 
%
% relevantTasks                  ... A list of task IDs for which training
%                                           patterns should be generated.
%                                           Tasks are enumerated according
%                                           the entries of a adjencency
%                                           matrix A of a bipartite graph of
%                                           size N, e.g.
%                                           If NPathways = 3,
%
%                                                 [ 1 2 3 ]
%                                           A =   [ 4 5 6 ]
%                                                 [ 7 8 9 ]
%
%                                           where the rows of A
%                                           correspond to input nodes
%                                           and the columns of A correspond
%                                           to output nodes.
% RETURN VALUES:
%
% inputSgl      ...A P-by-I matrix that corresponds to the 
%                      single-task training set for input stimuli. The number
%                      of rows P of inputSgl corresponds to the number of
%                      training patterns. Each row corresponds to a
%                      particular stimulus that consists of NPathway
%                      stimulus dimensions with NFeatures units each. Each
%                      stimulus pattern is encoded among I = NPathway *
%                      NFeatures units. The value of each unit of a stimulus 
%                      dimension is a sample drawn from a Gaussian distribution
%                      N(mu_i, sigma_i), i ? {1, ..., NFeatures}, where mu_i ? {0,1}
%                      and sigma_i is sampled from the uniform distribution U[0, sdScale]. 
%                      By letting sum_i(mu_i) = 1, it is enforced that one unit per 
%                      stimulus dimension dominates for each stimulus.
%
% tasksSgl      ...A P-by-T matrix that corresponds to the 
%                      single-task training set for input tasks. The number of
%                       rows of tasksSgl corresponds to the number of
%                       training patterns P and the number of columns of
%                       tasksSgl corresponds to the total of possible
%                       tasks T, i.e. T = NPathways^2
%                       In single-task training a given row in tasksSgl
%                       denotes a one-hot vector indicating the current
%                       task to be executed.
%
%
% trainSgl      ...A P-by-R matrix that corresponds to the 
%                     single-task training set for correct network outputs.
%                     The number of rows of trainSgl corresponds to the
%                     number of training patterns P. Each row corresponds to a
%                      particular respnse that consists of NPathway
%                      output dimensions with NFeatures response units each.
%                      Each output pattern is encoded among R = NPathway *
%                      NFeatures units. For a given single task, only one
%                      response dimension is relevant while all other
%                      response dimensions are supressed (set to 0). In for
%                      each relevant response dimension only one output
%                      unit is allowed to be active (set to 1), namely the
%                      one that corresponds to the active feature of the
%                      relevant stimulus input dimension (according to the
%                      current task).
%
% tasksIdxSgl   ...1-by-P vector that indexes all tasks across all training patterns. 
%
% stimIdxSgl    ...1-by-P vector that indexes individual stimuli across all training patterns
%
% inputSgl_mask     ...corresponds to a full training set of inputSgl
%                              without repetitions the value of each unit of a 
%                              stimulus is ? {0,1}.
%
% tasksSgl_mask     ...corresponds to a full training set of tasksSgl
%                              without repetitions
%
% trainSgl_mask     ...corresponds to a full training set of trainSgl
%                              without repetitions
%
% multiCap             ... stores all possible* multitasking combinations 
%                               for each cardinality. E.g. multiCap{2}.input, multiCap{2}.tasks, 
%                               multiCap{2}.train are the training patterns for performing 
%                               2 tasks in parallel (for all possible combinations).
%                               multiCap only contains multitask conditions in which all simultaneously 
%                               executed tasks rely on different input and
%                               output patterns.

currFeaturesDims = 1:NPathways;

% additional argument may specify the relevant tasks being tested
if(~isempty(varargin))
    relevantTasks = varargin{1};
else
    relevantTasks = 1:(NPathways^2);
end

% create the task environment for performing just one task (single task)

% loop through each new training set
input_sglT = [];
tasks_sglT = [];
train_sglT = [];
tasksIdx_sglT = [];
stimIdx_sglT = [];


%% generate raw samples for each task

% if samples = [], then pair each task with every possible stimulus
if(isempty(samples))

    stimCombs = [1:NFeatures];
    for i = 2:NPathways
        stimCombs = combvec(stimCombs,[1:NFeatures]); 
    end

    for currTIdx = 1:length(relevantTasks)

        % set task
        currT = relevantTasks(currTIdx);

        % build task input
        currTasks = zeros(1,NPathways*NPathways);
        currTasks(currT) = 1;
        currTasks = repmat(currTasks(:)', size(stimCombs,2), 1); % backtransform: reshape(currTasks(1,:),10,10)'

        currTasksM = reshape(currTasks(1,:),NPathways,NPathways)';

        % build feature input
        currInput = zeros(size(stimCombs,2),NPathways*NFeatures);

        for i = 1:size(currInput,1)
            currInput(i,(currFeaturesDims-1).*(NFeatures)+stimCombs(:,i)') = 1;
        end

        % build output
        currTrain = zeros(size(stimCombs,2),NPathways*NFeatures);

        [relevantInput,relevantOutput] = find(currTasksM ==1);
        for i = 1:size(currInput,1)
            currTrain(i,[((relevantOutput-1)*NFeatures+1):((relevantOutput-1)*NFeatures+NFeatures)]) = currInput(i,((relevantInput-1)*NFeatures+1):((relevantInput-1)*NFeatures+NFeatures));
        end

        % build full training set
        tasks_sglT = [tasks_sglT; currTasks];
        input_sglT = [input_sglT; currInput];
        train_sglT = [train_sglT; currTrain];

        curr_tasksIdx = currT*ones(size(stimCombs,2),1);
        curr_stimIdx = transpose(1:size(stimCombs,2));
        tasksIdx_sglT = [tasksIdx_sglT; curr_tasksIdx];
        stimIdx_sglT = [stimIdx_sglT; curr_stimIdx];

    end

    
else % otherwise restrict number of stimuli per task by number of samples
     
     for currTIdx = 1:length(relevantTasks)

        % set task
        currT = relevantTasks(currTIdx);

        % build task input
        currTasks = zeros(1,NPathways*NPathways);
        currTasks(currT) = 1;
        currTasks = repmat(currTasks(:)', samples, 1); % backtransform: reshape(currTasks(1,:),10,10)'

        currTasksM = reshape(currTasks(1,:),NPathways,NPathways)';

        % no need to counterbalance feature input and output
        currInput = zeros(samples,NPathways*NFeatures);
        currTrain = zeros(samples,NPathways*NFeatures);
        
        % build full training set
        tasks_sglT = [tasks_sglT; currTasks];
        input_sglT = [input_sglT; currInput];
        train_sglT = [train_sglT; currTrain];

        curr_tasksIdx = currT*ones(samples,1);
        curr_stimIdx = transpose(1:samples);
        tasksIdx_sglT = [tasksIdx_sglT; curr_tasksIdx];
        stimIdx_sglT = [stimIdx_sglT; curr_stimIdx];

     end
     
end

%% add gaussian noise to single task patterns

inputSgl_mask = input_sglT;
tasksSgl_mask = tasks_sglT;
trainSgl_mask = train_sglT;
tasksIdxSgl_mask = tasksIdx_sglT;
stimIdxSgl_mask = stimIdx_sglT;

% fill in with multi-variate gaussian samples
 
inputSgl = nan(size(inputSgl_mask));
tasksSgl = tasksSgl_mask;
trainSgl = trainSgl_mask;
tasksIdxSgl = tasksIdxSgl_mask;
stimIdxSgl = stimIdxSgl_mask;

if(isempty(samples))
    nStims = size(stimCombs,2);
else
    nStims = samples;
end
sd = rand(NFeatures, NFeatures)*sdScale;


inputSgl_tmp = nan(size(inputSgl_mask));

for currTIdx = 1:length(relevantTasks)

    if(currTIdx == 1 || ~sameStimuliAcrossTasks)
        for inputIdx = 1:nStims;
            
            rowIdx = (currTIdx-1)*nStims+inputIdx;
            
            % if samples are random, generate them first
            if(~isempty(samples))
                % generate stimulus
                stimCombs = randsample(NFeatures, NPathways, true);
                inputSgl_mask(rowIdx,(currFeaturesDims-1).*(NFeatures)+stimCombs(:)') = 1;
                
                % compute correct training pattern
                currTasks = tasksSgl_mask(rowIdx,:);
                currTasks = repmat(currTasks(:)', samples, 1); % backtransform: reshape(currTasks(1,:),10,10)'
                currTasksM = reshape(currTasks(1,:),NPathways,NPathways)';
                [relevantInput,relevantOutput] = find(currTasksM ==1);
                trainSgl(rowIdx,[((relevantOutput-1)*NFeatures+1):((relevantOutput-1)*NFeatures+NFeatures)]) = inputSgl_mask(rowIdx,((relevantInput-1)*NFeatures+1):((relevantInput-1)*NFeatures+NFeatures));
            end

            for dimensionIdx = 1:NPathways
                  colIdx = (NFeatures*(dimensionIdx-1)+1):(NFeatures*dimensionIdx);
                  mu = inputSgl_mask(rowIdx, colIdx);
                  feature = find(mu==1);
                  x = mvnrnd(mu,sd(feature,:), 1); 

                  inputSgl_tmp(rowIdx, colIdx) = x;
            end

        end
    else
        inputSgl_tmp(((currTIdx-1)*nStims+1):(currTIdx*nStims),:) = inputSgl_tmp(1:nStims,:);

        % if samples are random generate correspoding output
        if(~isempty(samples))
            for inputIdx = 1:nStims;

                rowIdx = (currTIdx-1)*nStims+inputIdx;

                % compute correct training pattern
                currTasks = tasksSgl_mask(rowIdx,:);
                currTasks = repmat(currTasks(:)', samples, 1); % backtransform: reshape(currTasks(1,:),10,10)'
                currTasksM = reshape(currTasks(1,:),NPathways,NPathways)';
                [relevantInput,relevantOutput] = find(currTasksM ==1);
                trainSgl(rowIdx,[((relevantOutput-1)*NFeatures+1):((relevantOutput-1)*NFeatures+NFeatures)]) = inputSgl_tmp(rowIdx,((relevantInput-1)*NFeatures+1):((relevantInput-1)*NFeatures+NFeatures));

            end
        end
        
    end

end

inputSgl = inputSgl_tmp;

%% create multitasking patterns

multiCap{1}.input = inputSgl;
multiCap{1}.tasks = tasksSgl;
multiCap{1}.train = trainSgl;
    
if(isempty(samples))
    samplesPerTask = size(stimCombs,2);
else
    samplesPerTask = samples;
end

numInputUnits = size(inputSgl,2);
numTaskUnits = size(tasksSgl,2);
numOutputUnits = size(trainSgl,2);

relevantTaskM = zeros(NPathways, NPathways);
relevantTaskM(relevantTasks) = 1;
relevantTaskM = transpose(relevantTaskM); % input components are rows, output components are columns

% Make sure the tasks are in order
myTasks = sort(relevantTasks);

for Nactive = 2:NPathways
    
    taskPaths = [zeros(1,NPathways-Nactive) 1:NPathways];
    taskCombs = taskPaths;

    for k = 2:1:NPathways
        taskCombs = combvec(taskCombs,taskPaths); % 1:NPathways
        for j = 1:(k-1)
            indicies = find(taskCombs(j,:) == taskCombs(k,:) & taskCombs(j,:) ~= 0);
            taskCombs(:,indicies) = [];
        end
    end

    % delete multitasking conditions that exceed specified number of active
    % tasks
    numNonZeroElements = taskCombs ~= 0;
    numNonZeroElements = sum(numNonZeroElements);
    taskCombs(:,numNonZeroElements ~= Nactive) = [];
    taskCombs = transpose(unique(transpose(taskCombs),'rows'));

    % I want to sort the tasks by row then column. They are encoded so that
    % values of 0 correspond to empty rows. I make these a large value so 
    % that they will be sorted last in the order of combinations. 
    taskCombs(taskCombs == 0) = 1e38;
    taskCombs = sortrows(taskCombs', 1:size(taskCombs,1))';
    taskCombs(taskCombs == 1e38) = 0;
    
    inputMulti = nan(samplesPerTask * size(taskCombs,2),numInputUnits);
    tasksMulti = nan(samplesPerTask * size(taskCombs,2),numTaskUnits);
    trainMulti = nan(samplesPerTask * size(taskCombs,2),numOutputUnits);

    myTasks = GenMultiTasks(NPathways, myTasks);
    
    taskCombCounter = 1;

    validMultiTasks = [];
    allMultiTasks = [];
    for currTaskComb = 1:size(taskCombs,2)

        % build task input
        currTasksM = zeros(NPathways, NPathways); % input components are rows, output components are columns

        for k = 1:NPathways;
            if(taskCombs(k,currTaskComb) ~= 0)
                currTasksM(k,taskCombs(k,currTaskComb)) = 1;
            end
        end

        allMultiTasks = [allMultiTasks; find(currTasksM' == 1)'];
        
        if(max(sum(currTasksM,1)) > 1) % hard constraint
            error('Overlapping tasks: Can''t use one output modality for two different feature dimensions');
        end
        if(max(sum(currTasksM,2)) > 1) % soft constraint (can potentially be removed)
            error('Overlapping feature dimensions: Can''t perform two tasks on the same input features');
        end
        
        % check if task combinations only contains relevant tasks
        if( mean(mean((relevantTaskM & currTasksM) == currTasksM)) == 1)

            validMultiTasks = [validMultiTasks; find(currTasksM' == 1)'];
                  
            currTasks = currTasksM';
            currTasks = repmat(currTasks(:)', samplesPerTask, 1); % backtransform: reshape(currTasks(1,:),10,10)'

            tasksMulti(((taskCombCounter-1)*samplesPerTask+1):(taskCombCounter*samplesPerTask),:) = currTasks;

            % build stimulus input
            inputSgl_tmp = nan(samplesPerTask, size(inputSgl,2));
            if(sameStimuliAcrossTasks)
                inputSgl_tmp = inputSgl(1:samplesPerTask,:);
            else
                for inputIdx = 1:nStims;
                    
                    rowIdx = inputIdx;
                    currentInputStim = zeros(1, size(inputSgl_mask,2));
                    % if stimuli are counterbalanced, then sample from mask
                    if(isempty(samples))
                        currentInputStim = inputSgl_mask(rowIdx, :);
                    else
                        % otherwise draw random sample for stimulus
                        stimCombs = randsample(NFeatures, NPathways, true);
                        currentInputStim((currFeaturesDims-1).*(NFeatures)+stimCombs(:)') = 1;
                    end

                    for dimensionIdx = 1:NPathways
                          
                          colIdx = (NFeatures*(dimensionIdx-1)+1):(NFeatures*dimensionIdx);
                          mu = currentInputStim(colIdx);
                          feature = find(mu==1);
                          x = mvnrnd(mu,sd(feature,:), 1); 
                          
                          inputSgl_tmp(rowIdx, colIdx) = x;
                    end

                end

            end
            inputMulti(((taskCombCounter-1)*samplesPerTask+1):(taskCombCounter*samplesPerTask),:) = inputSgl_tmp;

            % build training output

            currTrain = zeros(samplesPerTask,NPathways*NFeatures);

            activeTasks = find(currTasks(1,:) == 1);
            for k = 1:length(activeTasks)

                task = activeTasks(k);

                taskTemplate = zeros(1, size(currTasks,2));
                taskTemplate(task) = 1;

                if(isempty(samples))
                    % correct outpattern already available
                    taskData = trainSgl(ismember(tasksSgl, taskTemplate, 'rows'),:);
                    currTrain = currTrain + taskData;
                else
                    % generate correct output pattern
                    for inputIdx = 1:nStims;
                        currInput = inputSgl_tmp(inputIdx, :);
                        currTasks = repmat(taskTemplate(:)', samples, 1); 
                        currTasksM = reshape(currTasks(1,:),NPathways,NPathways)';
                        [relevantInput,relevantOutput] = find(currTasksM ==1);
                        taskData = zeros(1, size(trainSgl, 2));
                        taskData(((relevantOutput-1)*NFeatures+1):((relevantOutput-1)*NFeatures+NFeatures)) = currInput(((relevantInput-1)*NFeatures+1):((relevantInput-1)*NFeatures+NFeatures));
                        
                        currTrain(inputIdx,:) = currTrain(inputIdx,:)  + taskData;
                    end
                end
            end

            trainMulti(((taskCombCounter-1)*samplesPerTask+1):(taskCombCounter*samplesPerTask),:) = currTrain;
            
            taskCombCounter = taskCombCounter + 1;
        
        end
        
    end
      
    % cut empty patterns 
    emptyRows = ((taskCombCounter-1)*samplesPerTask+1) : size(inputMulti, 1);
    inputMulti(emptyRows, :) = [];
    tasksMulti(emptyRows, :) = [];
    trainMulti(emptyRows, :) = [];
    
    checkEqual = isequal( validMultiTasks, myTasks) | ...
        (size(validMultiTasks,1) == 0 & size(myTasks,1) == 0);
    
    assert(checkEqual); 
    
    multiCap{Nactive}.input = inputMulti;
    multiCap{Nactive}.tasks = tasksMulti;
    multiCap{Nactive}.train = trainMulti;
end    
    

end


