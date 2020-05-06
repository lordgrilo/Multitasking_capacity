function [inputSgl, tasksSgl, trainSgl, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask multiCap] = createSingleTaskPatterns(NPathways, NFeatures, samples, sdScale, sameStimuliAcrossTasks, varargin)
% createsingleTaskPatterns generate single tasking patterns
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

currFeaturesDims = 1:NPathways;

% additional argument may specify the relevant tasks being tested
if(~isempty(varargin))
    relevantTasks = varargin{1};
else
    relevantTasks = 1:(NPathways^2);
end

% Force relevantTasks to be a column vector, some of our code assumes this.
if(size(relevantTasks, 1) == 1)
    relevantTasks = relevantTasks';
end

% create the task environment for performing just one task (single task)

% Pre-allocate matrices for input, task, and training signal. To do this, we
% first need to figure out how many samples we will have for each task. If
% the user has not specified it, then:
if(~isempty(samples))
    SAMPLES_PER_TASK = samples;
else
    % If the user didn't specify how many samples, then assume we want to
    % do all possible stimulus inputs per task.
    SAMPLES_PER_TASK = NFeatures^NPathways;
end

% Get the total number of training samples
TOTAL_SAMPLES = length(relevantTasks)*SAMPLES_PER_TASK;

% Pre-allocate our task, input, and training signal matrices.
tasks_sglT = zeros(TOTAL_SAMPLES, NPathways*NPathways);
input_sglT = zeros(TOTAL_SAMPLES, NPathways*NFeatures);
train_sglT = zeros(TOTAL_SAMPLES, NPathways*NFeatures);

% Generate the task layer inputs. These are just one-hot encoded vector
% of the task index, repeated SAMPLES_PER_TASK time for each.
rTasks = repmat(relevantTasks', SAMPLES_PER_TASK, 1);
tasks_sglT( sub2ind(size(tasks_sglT), 1:size(tasks_sglT,1), rTasks(:)') ) = 1;

% Generate a matrix that specifies the task index for each sample
tasksIdx_sglT = repmat(relevantTasks', SAMPLES_PER_TASK, 1); tasksIdx_sglT = tasksIdx_sglT(:);

% Generate a matrix that specifies the sample\stim index within each task's
% group.
stimIdx_sglT = repmat(transpose(1:SAMPLES_PER_TASK), [length(relevantTasks) 1]);

% Convert linear tasks indices to input\output row\column subscripts.
% Matlab uses row major indexing so we need to switch the outputs arounnd
[relevantOutputs, relevantInputs] = ind2sub([NPathways NPathways], relevantTasks);

% if samples = [], then pair each task with every possible stimulus
if(isempty(samples))

    % We start by creating all possible stimulus input combinations. 
    stimCombs = 1:NFeatures;
    for i = 2:NPathways
        stimCombs = combvec(stimCombs,1:NFeatures); 
    end

    % Create a sparse representation of these stimuli as one-hot encoded
    % vectors.
    task_input = reshape(ind2vec(stimCombs(:)'), [NFeatures*NPathways, SAMPLES_PER_TASK])';
    
    % Repeat this for each task
    input_sglT = repmat(task_input, [length(relevantTasks), 1]);
       
    % Go through each relevant task, and map the appropriate inputs from
    % the stimulus to the the appropriate ouput in the training signal.
    
    sampleIdx = 1;
    for currTIdx = 1:length(relevantTasks)
        
        sampleStartIdx = sampleIdx;
        
        % Copy portions of each training stimulus' based on the current task. 
        currTrain = zeros(SAMPLES_PER_TASK, NPathways*NFeatures);
        for i = 1:SAMPLES_PER_TASK
            currTrain(i,((relevantOutputs(currTIdx)-1)*NFeatures+1):((relevantOutputs(currTIdx)-1)*NFeatures+NFeatures)) = input_sglT(sampleIdx,((relevantInputs(currTIdx)-1)*NFeatures+1):((relevantInputs(currTIdx)-1)*NFeatures+NFeatures));
            sampleIdx = sampleIdx + 1;
        end

        % build full training set
        train_sglT(sampleStartIdx:(sampleStartIdx+SAMPLES_PER_TASK-1),:) = currTrain;
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

sd = rand(NFeatures, NFeatures)*sdScale;

inputSgl_tmp = nan(size(inputSgl_mask));

for currTIdx = 1:length(relevantTasks)
    
    relevantOutput = relevantOutputs(currTIdx);
    relevantInput = relevantInputs(currTIdx);
    
    if(currTIdx == 1 || ~sameStimuliAcrossTasks)
        for inputIdx = 1:SAMPLES_PER_TASK
            
            rowIdx = (currTIdx-1)*SAMPLES_PER_TASK+inputIdx;
            
            % if samples are random, generate them first
            if(~isempty(samples))
                % generate stimulus
                stimCombs = randsample(NFeatures, NPathways, true);
                inputSgl_mask(rowIdx,(currFeaturesDims-1).*(NFeatures)+stimCombs(:)') = 1;
                
                % compute correct training pattern
                trainSgl(rowIdx,[((relevantOutput-1)*NFeatures+1):((relevantOutput-1)*NFeatures+NFeatures)]) = inputSgl_mask(rowIdx,((relevantInput-1)*NFeatures+1):((relevantInput-1)*NFeatures+NFeatures));
            end

            for dimensionIdx = 1:NPathways
                  colIdx = (NFeatures*(dimensionIdx-1)+1):(NFeatures*dimensionIdx);
                  mu = inputSgl_mask(rowIdx, colIdx);
                  x = mvnrnd(mu,sd(mu==1,:), 1); 
                  inputSgl_tmp(rowIdx, colIdx) = x;
            end

        end
    else
        inputSgl_tmp(((currTIdx-1)*SAMPLES_PER_TASK+1):(currTIdx*SAMPLES_PER_TASK),:) = inputSgl_tmp(1:SAMPLES_PER_TASK,:);

        % if samples are random generate correspoding output
        if(~isempty(samples))
            for inputIdx = 1:SAMPLES_PER_TASK;
                rowIdx = (currTIdx-1)*SAMPLES_PER_TASK+inputIdx;

                % compute correct training pattern
                trainSgl(rowIdx,((relevantOutput-1)*NFeatures+1):((relevantOutput-1)*NFeatures+NFeatures)) = inputSgl_tmp(rowIdx,((relevantInput-1)*NFeatures+1):((relevantInput-1)*NFeatures+NFeatures));
            end
        end
        
    end

end

inputSgl = inputSgl_tmp;

% If the caller has requested the multitasking patterns, generate those as
% well.
if(nargout > 8)

    multiCap = cell(1, NPathways);
    multiCap{1}.taskCombs = relevantTasks;
    multiCap{1}.input = inputSgl;
    multiCap{1}.tasks = tasksSgl;
    multiCap{1}.train = trainSgl;

    % Create matrix to keep track of which rows of tasks correspond to
    % which rows of taskCombs.
    taskIdx = repmat(1:length(relevantTasks), [SAMPLES_PER_TASK,1]); taskIdx = taskIdx(:);
    multiCap{1}.taskIdx = taskIdx;
    
    samplesPerTask = SAMPLES_PER_TASK;
    
    numInputUnits = size(inputSgl,2);
    numTaskUnits = size(tasksSgl,2);
    numOutputUnits = size(trainSgl,2);

    taskCombs = sort(relevantTasks);

    for Nactive = 2:NPathways

        taskCombs = GenMultiTasks(NPathways, taskCombs);
        NUM_MULTI_TASKS = size(taskCombs,1);
        
        inputMulti = nan(samplesPerTask * NUM_MULTI_TASKS,numInputUnits);
        tasksMulti = nan(samplesPerTask * NUM_MULTI_TASKS,numTaskUnits);
        trainMulti = nan(samplesPerTask * NUM_MULTI_TASKS,numOutputUnits);

        taskCombCounter = 1;

        % Keep track of which tasks have congruent or incongruent outputs
        congruency = zeros(samplesPerTask * NUM_MULTI_TASKS, 1);
         
        for currTaskComb = 1:NUM_MULTI_TASKS
            row_range = ((taskCombCounter-1)*samplesPerTask+1):(taskCombCounter*samplesPerTask);
            
            % build task input
            currTasksM = zeros(NPathways, NPathways); % input components are rows, output components are columns
            currTasksM(taskCombs(currTaskComb, :)) = 1;
            
            if(max(sum(currTasksM,1)) > 1) % hard constraint
                error('Overlapping tasks: Can''t use one output modality for two different feature dimensions');
            end
            if(max(sum(currTasksM,2)) > 1) % soft constraint (can potentially be removed)
                error('Overlapping feature dimensions: Can''t perform two tasks on the same input features');
            end

            currTasks = currTasksM;
            currTasks = repmat(currTasks(:)', samplesPerTask, 1); % backtransform: reshape(currTasks(1,:),10,10)'

            tasksMulti(((taskCombCounter-1)*samplesPerTask+1):(taskCombCounter*samplesPerTask),:) = currTasks;

            % build stimulus input
            inputSgl_tmp = nan(samplesPerTask, size(inputSgl,2));
            if(sameStimuliAcrossTasks)
                inputSgl_tmp = inputSgl(1:samplesPerTask,:);
            else
                for inputIdx = 1:SAMPLES_PER_TASK;

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
                          x = mvnrnd(mu,sd(mu==1,:), 1); 

                          inputSgl_tmp(rowIdx, colIdx) = x;
                    end

                end

            end
            inputMulti(((taskCombCounter-1)*samplesPerTask+1):(taskCombCounter*samplesPerTask),:) = inputSgl_tmp;

            % build training output
            currTrain = zeros(samplesPerTask,NPathways*NFeatures);

            % We want to keep track, for each sample, how many times a
            % feature dimension is used in the output. This will let us
            % determine whether the sample is congruent or incongruent
            featureDimCount = zeros(samplesPerTask, NFeatures); 
            
            activeTasks = find(currTasks(1,:) == 1);
            for k = 1:length(activeTasks)

                task = activeTasks(k);

                taskTemplate = zeros(1, size(currTasks,2));
                taskTemplate(task) = 1;
                currTasksM = reshape(taskTemplate(:)',NPathways,NPathways)';
                [relevantInput,relevantOutput] = find(currTasksM ==1);
                feature_dims_out = ((relevantOutput-1)*NFeatures+1):((relevantOutput-1)*NFeatures+NFeatures);
                feature_dims_in = ((relevantInput-1)*NFeatures+1):((relevantInput-1)*NFeatures+NFeatures);
                
                if(isempty(samples))
                    % correct outpattern already available
                    taskData = trainSgl(ismember(tasksSgl, taskTemplate, 'rows'),:);
                    currTrain = currTrain + taskData;
                else                        
                    % generate correct output pattern
                    currTrain(:, feature_dims_out) = currTrain(:, feature_dims_out) + inputSgl_tmp(:, feature_dims_in);
                end
                
                featureDimCount = featureDimCount + currTrain(:,feature_dims_out);
            end

            % Copy the training signal
            trainMulti(row_range,:) = currTrain;

            % Mark congruent tasks, otherwise they are incongruent
            numCorrectOutputUnits = sum(currTrain, 2);
            congruency(row_range(max(featureDimCount, [], 2) == numCorrectOutputUnits)) = 1;
            
            taskCombCounter = taskCombCounter + 1;
        end
        
        % Create matrix to keep track of which rows of tasks correspond to
        % which rows of taskCombs.
        taskIdx = repmat(1:NUM_MULTI_TASKS, [samplesPerTask,1]); taskIdx = taskIdx(:);
        
        multiCap{Nactive}.taskCombs = taskCombs;
        multiCap{Nactive}.taskIdx = taskIdx;
        multiCap{Nactive}.congruency = logical(congruency);
        multiCap{Nactive}.input = inputMulti;
        multiCap{Nactive}.tasks = tasksMulti;
        multiCap{Nactive}.train = trainMulti;
    end     
    
end
