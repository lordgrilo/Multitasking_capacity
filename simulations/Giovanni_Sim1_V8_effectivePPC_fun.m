function [outfile] = Giovanni_Sim1_V8_effectivePPC_fun(inputArg, hiddenArg, silence)

addpath('/home/dmturner/RTDist_1.0.0-rc2/MATLAB');

hiddenArg = str2num(hiddenArg);
nHidden = hiddenArg;
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


% TODO: 
% - when generating graphs, just give it a fixed probability
% - count proportion of tasks sets in which all tasks are above threshold


    if(nargin < 2)
        silence = 0;
    end

    function my_printf(varargin)  
        if(~silence)
           fprintf(1, varargin{:});
        end
    end

%     rng(0);

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
    correlation_thresholds = [0:0.025:1];

    %% generate folder list and set current folder
    N_list = [4 5 6 7 8 9];
    graphSpecsFolder_list = {};
    
    densityList = [0.2];

    folderCounter = 1;
    for N_idx = N_list
        for overlap_idx = 1:length(densityList);
            density = num2str(densityList(overlap_idx));
            graphSpecsFolder_list{folderCounter} = strcat('N_', num2str(N_idx), '_z_', density);
            folderCounter = folderCounter+1;
        end
    end

    if(isa(inputArg, 'char'))
        numGraphsPerSpec = 40;
        graphFolder = 'GioGraphGeneration/shared/edgelists_fixedDensity';
        inputArgDouble = str2double(inputArg);
        graphSpecIdx = ceil(inputArgDouble/numGraphsPerSpec);
        disp(['graphSpecIdx: ' num2str(graphSpecIdx)]);
        disp(graphSpecsFolder_list);

        graphSpecsFolder = graphSpecsFolder_list{graphSpecIdx};

        samplingFolder = 'ER';

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

        A_N = length(graphs);
        A_ID =mod(inputArgDouble-1, A_N) + 1;

        %% run simulation loop

        % set up log
        MISLog_full = nan(1, replications);

        % get graph
        A = graphs{A_ID}.A;
    else
        A = inputArg;
    end
    
    % set number of features per dimension
    if(~fixNFeatures)
        NFeatures = min(size(A));
    else
        NFeatures = NFeaturesFixed;
    end

    % compute dependency graph
    A_dual = getDependencyGraph(A);

    % compute MIS of dependency graph
%     m = nnz(A);
%     MultiTask= sum(findMIS(logical(A_dual),[1:m]));
    [MultiTask] = getMISFromBipartiteGraph(A);
    MISLog = MultiTask;
    
    %% check if log file already exists
    
    outfile = ['logfiles/Giovanni_Simulation_V8_h' num2str(nHidden) '_' graphSpecsFolder '_' samplingFolder '_A' num2str(A_ID) '_' num2str(log_version) '.mat'];
    if exist(outfile, 'file') == 2
        return;
    end

    %% create full task environment
    samplesPerTask = samplesPerTask_train;

    NPathways = max(size(A));

    [input, tasks, train] = generateEnvironmentToGraph(A, NFeatures, samplesPerTask, sdScale, sameStimuliAcrossTasks);

    my_printf('Generated task environment...\n');

    for rep = 1:replications

        my_printf('repetition: %d\n', rep);

        % compute MIS of dependency graph
%         m = nnz(A);
        tic
%         MultiTask= sum(findMIS(logical(A_dual),[1:m]));
        [MultiTask] = getMISFromBipartiteGraph(A);
        timing.findMIS(rep) = toc;
        MISLog_full(1, rep) = str2double(MultiTask);

        % build network
        taskNet = NNmodel(nHidden, learningRate, bias, init_scale, thresh, decay, hiddenPathSize, outputPathSize);
        taskNet.NPathways = NPathways;
        taskNet.silence = silence;
        
        % initialize network according to bipartite graph
        taskNet.setData(input, tasks, train);
        taskNet.configure(); 
        taskNet = initializeNetworkToGraph(A, taskNet, 0, 1);

        %% train network on all specified single tasks
        tic
        MSE_log = zeros(1, iterations_Train);
        for iter = 1:iterations_Train

            % sample training data
            samplesPerTask = samplesPerTask_train;
            [input, tasks, train] = generateEnvironmentToGraph(A, NFeatures, samplesPerTask, sdScale, sameStimuliAcrossTasks);

            % train on data
            taskNet.trainOnline(1, input, tasks, train);

            % is performance criterion reached?
            MSE_log(iter) = taskNet.MSE_log(end);
            if (taskNet.MSE_log(end) <= taskNet.thresh)
                break
            end
            my_printf('\t%d\t%f\n', iter, taskNet.MSE_log(end));

        end
        timing.trainOnline(rep) = toc;

        %% test network performance
        my_printf('Generating multi-task patterns ... ');
        samplesPerTask = samplesPerTask_test;
        [input, tasks, train, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask, multiCap, multiCap_con, multiCap_inc, relevantTasks, NPathways] = generateEnvironmentToGraph(A, NFeatures, samplesPerTask, sdScale, sameStimuliAcrossTasks);
        my_printf('Done.\n');

        % compute optimal control policy
        [taskCardinality, maximumPerformanceCurve, minimumPerformanceCurve, effectivePPC, effectivePPC_unthresholded, timing_policy, effectivePPC_permissive] = findOptimalControlPolicy_4(taskNet, NPathways, multiCap_inc, goodPerformanceThresholds);

        % log data
        taskCardinalityLog.absoluteAccuracy(rep, :) = taskCardinality.absoluteAccuracy;
        taskCardinalityLog.PCorrect(rep, :) = taskCardinality.PCorrect;
        taskCardinalityLog.LCA(rep, :) = taskCardinality.LCA;
        
        maximumPerformanceCurveLog.absoluteAccuracy(rep,:) = maximumPerformanceCurve.absoluteAccuracy;
        maximumPerformanceCurveLog.PCorrect(rep,:) = maximumPerformanceCurve.PCorrect;
        maximumPerformanceCurveLog.LCA(rep,:) = maximumPerformanceCurve.LCA;

        minimumPerformanceCurveLog.absoluteAccuracy(rep,:) = minimumPerformanceCurve.absoluteAccuracy;
        minimumPerformanceCurveLog.PCorrect(rep,:) = minimumPerformanceCurve.PCorrect;
        minimumPerformanceCurveLog.LCA(rep,:) = minimumPerformanceCurve.LCA;
        
        effectivePPCLog.absoluteAccuracy(rep,:,:) = effectivePPC.absoluteAccuracy;
        effectivePPCLog.PCorrect(rep,:,:) = effectivePPC.PCorrect;
        effectivePPCLog.LCA(rep,:,:) = effectivePPC.LCA;
        
        effectivePPCLog_permissive.absoluteAccuracy(rep,:,:) = effectivePPC_permissive.absoluteAccuracy;
        effectivePPCLog_permissive.PCorrect(rep,:,:) = effectivePPC_permissive.PCorrect;
        effectivePPCLog_permissive.LCA(rep,:,:) = effectivePPC_permissive.LCA;
        
        effectivePPCLog_unthresholded.absoluteAccuracy(rep,:,:) = effectivePPC_unthresholded.absoluteAccuracy;
        effectivePPCLog_unthresholded.PCorrect(rep,:,:) = effectivePPC_unthresholded.PCorrect;
        effectivePPCLog_unthresholded.LCA(rep,:,:) = effectivePPC_unthresholded.LCA;

        timing.findOptimalControlPolicy_absoluteAccuracy(rep) = timing_policy.findOptimalControlPolicy_absoluteAccuracy;
        timing.findOptimalControlPolicy_PCorrect(rep) = timing_policy.findOptimalControlPolicy_PCorrect;
        timing.findOptimalControlPolicy_LCA(rep) = timing_policy.findOptimalControlPolicy_LCA;

        %% get similariy metric for hidden and output task representations
        tic
        [hiddenSimilarity, outputSimilarity] = computeTaskSimilarity(taskNet, input, tasks);
        timing.extractMIS(rep) = toc;

        % extract MIS from neural network
        tic
        for corr_Idx = 1:length(correlation_thresholds)
            corr_threshold = correlation_thresholds(corr_Idx);
            [~, maxCarryingCapacity] = getMaxCarryingCapacity(hiddenSimilarity, outputSimilarity, corr_threshold);
            extractedMISLog.MIS(rep, corr_Idx) = str2num(maxCarryingCapacity);
        end
        timing.extractMIS(rep) = timing.extractMIS(rep) + toc/length(correlation_thresholds);

        timing.findMIS(rep) = timing.findMIS(rep);
        timing.trainOnline(rep) = timing.trainOnline(rep);
        timing.extractMIS(rep) = timing.extractMIS(rep);
        A_MIS(rep) = MISLog_full(1, rep);
        extractedMIS(rep, :) = extractedMISLog.MIS(rep, :);
        NFeatures(rep) = NFeatures;

        my_printf('repetition %d/%d\n', rep, replications);
    end

    if(isa(inputArg, 'char'))
        outfile = ['logfiles/Giovanni_Simulation_V8_h' num2str(nHidden) '_' graphSpecsFolder '_' samplingFolder '_A' num2str(A_ID) '_' num2str(log_version) '.mat'];
        save(outfile);
    end
end