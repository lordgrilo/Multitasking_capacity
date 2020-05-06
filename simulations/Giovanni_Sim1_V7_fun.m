function Giovanni_Sim1_V7_fun(inputArg)

%% description
% 
% Simulation # 1
% 
% The simulation involves initializing a nonlinear neural network according
% to a bipartite graph. The bipartite graph specifies the task environment
% (how many input and output components), as well as the relationship
% between task representations in the neural network (i.e. if two tasks
% rely on the same representations at the hidden layer). 
% After the network is initialized, it is trained on the set of tasks that
% is specified by the bipartite graph. ITraining is restricted to weights
% projecting to the output layer of the network.
% Once the network is trained, the optimal amount of tasks that can be
% performed in parallel is computed according to a specified value
% function.
%
% author: Sebastian Musslick


%% initialization
% clc;
% clear all;

 % specify number of available cores
% numCores = 2;              
% maxNumCompThreads(numCores);

% update meta simulation parameters
log_version = 2;
replications = 1;                  %  replications per simulation

% set up network parameters
init_scale = 0.1;                  % scales for initialized random weights 
learningRate = 0.3;             % learning rate
decay = 0.0000;                 % weight penalization parameter
bias = -5;                          % weight from bias units to hidden & output units
iterations_Train = 10000;      % maximum number of training iterations
thresh = 0.001;                  % mean-squared error stopping criterion

hiddenPathSize = 1;             % group size of hidden units that receive the same weights from the task layer)
outputPathSize = 1;              % group size of output units that receive the same weights from the task layer)

% training environment parameters
subSampleGraphs = 1;            % indicates whether to only test subsamples of all possible graphs
NSubsamples = 5;                    % size of tested graph subsamples for every MIS value
sdScale = 0;
sameStimuliAcrossTasks = 1;      % use same stimuli across tasks? (this parameter is only relevant if sdScale > 0)
samplesPerTask_train = 100; % stimuli per task for training sets
samplesPerTask_test = 1000; % stimuli per task for testing sets

fixNFeatures = 0;
NFeaturesFixed = 6;

goodPerformanceThresholds = [0.1:0.05:0.95 0.99];
correlation_thresholds = [0.1:0.05:0.9];

%% generate folder list and set current folder
N_list = [4 5 6];
graphSpecsFolder_list = {};

folderCounter = 1;
for N_idx = N_list
    for overlap_idx = 1:(N_idx-1)
        graphSpecsFolder_list{folderCounter} = strcat('N_', num2str(N_idx), '_z_', num2str(overlap_idx));
        folderCounter = folderCounter+1;
    end
end

graphFolder = 'GioGraphGeneration/shared/edgelists';
inputArgDouble = str2double(inputArg);
graphSpecIdx = ceil(inputArgDouble/4);

graphSpecsFolder = graphSpecsFolder_list{graphSpecIdx};
subFolders = {'CL', 'CM', 'ER', 'regular'};

subFolderIdx = mod(inputArgDouble-1,4)+1;

    samplingFolder = subFolders{subFolderIdx};

    %% read graphs from folder

    fullFolder = [graphFolder '/' graphSpecsFolder '/' samplingFolder];

    files = dir(fullFolder);

    filenames = {};
    fileCounter = 1;

    for fileIdx = 1:length(files)

        % get file name
        fileName = files(fileIdx).name;

        % only proceed if file is an *.edges file
        if(length(fileName) >7)

            if(strcmp(fileName((end-5):end), '.edges'))

                % store file name
                graphs{fileCounter}.fileName = fileName;

                % read in graph
                filepath = [fullFolder '/' fileName];
                graphs{fileCounter}.A = getGraphFromEdgeList(filepath);

                % don't process empty graphs
                if(~isempty(graphs{fileCounter}.A))
                    filenames{fileCounter} = fileName;
                    fileCounter = fileCounter + 1;
                end
            end

        end

    end

    %% run simulation loop
    A_N = length(graphs);

    taskCardinalityLog_meanAcc = nan(A_N, replications, length(goodPerformanceThresholds));
    taskCardinalityLog_respProb = nan(A_N, replications, length(goodPerformanceThresholds));

    MISLog_full = nan(A_N, replications);
    MISLog = nan(A_N, 1);

    tic
    for A_ID = 1:length(graphs)

        A = graphs{A_ID}.A;
        if(~fixNFeatures)
            NFeatures = min(size(A));
        else
            NFeatures = NFeaturesFixed;
        end

        % compute dependency graph
        A_dual = getDependencyGraph(A);

        % compute MIS of dependency graph
        m = nnz(A);
        MultiTask= sum(findMIS(logical(A_dual),[1:m]));
        MISLog(A_ID) = MultiTask;

        % create full task environment
        samplesPerTask = samplesPerTask_train;

        [input, tasks, train, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask, multiCap, multiCap_con, multiCap_inc, relevantTasks, NPathways] = generateEnvironmentToGraph(A, NFeatures, samplesPerTask, sdScale, sameStimuliAcrossTasks);

        disp('Generated task environment...');

        for rep = 1:replications

            disp(strcat('repetition: ',num2str(rep)));

            % compute MIS of dependency graph
            m = nnz(A);
            tic
            MultiTask= sum(findMIS(logical(A_dual),[1:m]));
            timing{A_ID}.findMIS(rep) = toc;
            MISLog_full(A_ID, rep) = MultiTask;

            % build network
            nHidden = size(input,2);
            taskNet = NNmodel(nHidden, learningRate, bias, init_scale, thresh, decay, hiddenPathSize, outputPathSize);
            taskNet.NPathways = NPathways;

            % initialize network according to bipartite graph
            taskNet.setData(input, tasks, train);
            taskNet.configure(); 
            taskNet = initializeNetworkToGraph(A, taskNet, 0, 1);

            % train network on all specified single tasks
            tic
            MSE_log = zeros(1, iterations_Train);
            for iter = 1:iterations_Train

                % sample training data
                samplesPerTask = samplesPerTask_train;
                [input, tasks, train, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask, multiCap, multiCap_con, multiCap_inc, relevantTasks, NPathways] = generateEnvironmentToGraph(A, NFeatures, samplesPerTask, sdScale, sameStimuliAcrossTasks);

                % train on data
                taskNet.trainOnline(1, input, tasks, train);

                % is performance criterion reached?
                MSE_log(iter) = taskNet.MSE_log(end);
                if (taskNet.MSE_log(end) <= taskNet.thresh)
                    break
                end
                disp([iter taskNet.MSE_log(end) ]);

            end
            timing{A_ID}.trainOnline(rep) = toc;

            % test network performance
            samplesPerTask = samplesPerTask_test;
            [input, tasks, train, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask, multiCap, multiCap_con, multiCap_inc, relevantTasks, NPathways] = generateEnvironmentToGraph(A, NFeatures, samplesPerTask, sdScale, sameStimuliAcrossTasks);

            for threshIdx = 1:length(goodPerformanceThresholds)
                tic
                goodPerformanceThresh =  goodPerformanceThresholds(threshIdx);
                [~, taskCardinality_meanAcc, ~, ~, taskCardinality_respProb] = findOptimalControlPolicy_3(taskNet, NPathways, multiCap_inc, goodPerformanceThresh);
                timing{A_ID}.findOptimalControlPolicy(rep, threshIdx) = toc;
                taskCardinalityLog{A_ID}.meanAcc(rep, threshIdx) = taskCardinality_meanAcc;
                taskCardinalityLog{A_ID}.respProb(rep, threshIdx) = taskCardinality_respProb;

                disp(['Tested performance threshold ' num2str(threshIdx) '/' num2str(length(goodPerformanceThresholds))]);

            end

            % get similariy metric for hidden and output task representations
            tic
            [hiddenSimilarity, outputSimilarity] = computeTaskSimilarity(taskNet, input, tasks);
            timing{A_ID}.extractMIS(rep) = toc;

            % extract MIS from neural network
            tic
            for corr_Idx = 1:length(correlation_thresholds)
                corr_threshold = correlation_thresholds(corr_Idx);
                [~, maxCarryingCapacity] = getMaxCarryingCapacity(hiddenSimilarity, outputSimilarity, corr_threshold);
                extractedMISLog{A_ID}.MIS(rep, corr_Idx) = maxCarryingCapacity;
            end
            timing{A_ID}.extractMIS(rep) = timing{A_ID}.extractMIS(rep) + toc/length(correlation_thresholds);

            graphs{A_ID}.timing.findMIS(rep) = timing{A_ID}.findMIS(rep);
            graphs{A_ID}.timing.trainOnline(rep) = timing{A_ID}.trainOnline(rep);
            graphs{A_ID}.timing.extractMIS(rep) = timing{A_ID}.extractMIS(rep);
            graphs{A_ID}.optimalControlPolicy_meanAcc(rep,:) =  taskCardinalityLog{A_ID}.meanAcc(rep, :);
            graphs{A_ID}.optimalControlPolicy_respProb(rep,:) =  taskCardinalityLog{A_ID}.respProb(rep, :);
            graphs{A_ID}.A_MIS(rep) = MISLog_full(A_ID, rep);
            graphs{A_ID}.extractedMIS(rep, :) = extractedMISLog{A_ID}.MIS(rep, :);
            graphs{A_ID}.NFeatures(rep) = NFeatures;

            disp(['repetition ' num2str(rep) '/' num2str(replications)]);
        end

    end
     toc

    save([fullFolder '/Giovanni_Simulation_V7_' graphSpecsFolder '_' samplingFolder '_' num2str(log_version)]);


toc

end