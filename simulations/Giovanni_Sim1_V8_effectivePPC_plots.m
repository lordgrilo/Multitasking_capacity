%%
clear all;
clc;

% dataset specification
datafilePrefix = 'Giovanni_Simulation_V8';
correlation_threshold = 0.8;
optimalPerformanceThreshold = 0.90;
graphSpecsFolder_selected = 'N_4_z_0.2';
samplingFolder_selected = 'ER'; % CM ER regular
graphsPerSpecFolder = 40;
maxN = 8;

logFolder = 'logfiles/';

% load data
files = dir(logFolder);

% get valid file names
validFileNames = {};
for i =1:length(files)
    % check if this is a desired data file
    if(~isempty(strfind(files(i).name, datafilePrefix)))
        validFileNames{end+1} = files(i).name;
    end
end
if(~isempty(validFileNames))
    numGraphs = length(validFileNames);
    % load every valid file
    for fileNameIdx = 1:length(validFileNames);
        disp(['loading ' validFileNames{fileNameIdx} '...']);
        load(strcat(logFolder, validFileNames{fileNameIdx}));
        graph_log(fileNameIdx).taskCardinalityLog = taskCardinalityLog;
        graph_log(fileNameIdx).maximumPerformanceCurveLog = maximumPerformanceCurveLog;
        graph_log(fileNameIdx).minimumPerformanceCurveLog = minimumPerformanceCurveLog;
        graph_log(fileNameIdx).effectivePPCLog_restrictive = effectivePPCLog;
        graph_log(fileNameIdx).effectivePPCLog_permissive = effectivePPCLog_permissive;
        graph_log(fileNameIdx).A_MIS = A_MIS;
        graph_log(fileNameIdx).N = taskNet.NPathways;
        graph_log(fileNameIdx).extractedMIS = extractedMIS;
        graph_log(fileNameIdx).NFeatures = NFeatures;
        graph_log(fileNameIdx).timing.findMIS = timing.findMIS;
        graph_log(fileNameIdx).timing.extractMIS = timing.extractMIS;
        graph_log(fileNameIdx).timing.trainOnline = timing.trainOnline;
        graph_log(fileNameIdx).timing.findOptimalControlPolicy_absoluteAccuracy = timing.findOptimalControlPolicy_absoluteAccuracy;
        graph_log(fileNameIdx).timing.findOptimalControlPolicy_PCorrect = timing.findOptimalControlPolicy_PCorrect;
        graph_log(fileNameIdx).timing.findOptimalControlPolicy_LCA = timing.findOptimalControlPolicy_LCA;
        graph_log(fileNameIdx).graphSpecsFolder = graphSpecsFolder;
        graph_log(fileNameIdx).samplingFolder = samplingFolder;
        fileName = validFileNames{fileNameIdx};
        graph_log(fileNameIdx).z = str2num(fileName(strfind(fileName,'z')+2));
        
        % compute true MIS
        
    end
else
    error('No valid file names found');
end

close all;

disp('DONE LOADING');

% find correlation threshold index 
corrThreshIdx = find(correlation_thresholds == correlation_threshold,1);
if(isempty(corrThreshIdx))
    warning('Requested correlation threshold for MIS extraction does not exist in data set');
end

% find optimal performance threshold index
optPerformanceThreshIdx = find(round(goodPerformanceThresholds*100) == round(optimalPerformanceThreshold*100),1);
if(isempty(optPerformanceThreshIdx))
    warning('Requested optimal performance threshold does not exist in data set');
end

% plot settings
fontSize_title = 14;
fontSize_gca = 14;
fontSize_xlabel = 14;
fontSize_ylabel = 14;
fontSize_legend = 14;

fontName = 'Helvetica';
markerSize = 38;
sem_width = 2;
sem_marker = '-';

lineWidth = 3;

colors = [253 120 21; ... % orange
              31 104 172; ... % blue
              44 155 37; ... % green
              0     0   0  ; ... % black
            142 142 142; ... % grey 
            255 255 255] / 255; % white 
        
cContrast1 = 1;
cContrast2 = 2;
cContrast3 = 3;
cSingle = 4;
cWeak = 5;
cWhite = 6;


% plot settings

% plot graph analysis results

close all;

numRepetitions = replications;
numCorrTresholds = length(correlation_thresholds);
numOptimalPerformanceThresholds = length(goodPerformanceThresholds);

% full graph data

taskCardinalityLog.absoluteAccuracy = nan(numGraphs, numRepetitions, numOptimalPerformanceThresholds);
taskCardinalityLog.PCorrect = nan(numGraphs, numRepetitions, numOptimalPerformanceThresholds);
taskCardinalityLog.LCA = nan(numGraphs, numRepetitions, numOptimalPerformanceThresholds);

maximumPerformanceCurveLog.absoluteAccuracy = nan(numGraphs, numRepetitions, maxN);
maximumPerformanceCurveLog.PCorrect = nan(numGraphs, numRepetitions, maxN);
maximumPerformanceCurveLog.LCA = nan(numGraphs, numRepetitions, maxN);

minimumPerformanceCurveLog.absoluteAccuracy = nan(numGraphs, numRepetitions, maxN);
minimumPerformanceCurveLog.PCorrect = nan(numGraphs, numRepetitions, maxN);
minimumPerformanceCurveLog.LCA = nan(numGraphs, numRepetitions, maxN);

effectivePPCLog_restrictive.absoluteAccuracy = nan(numGraphs, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_restrictive.PCorrect = nan(numGraphs, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_restrictive.LCA = nan(numGraphs, numRepetitions, maxN, numOptimalPerformanceThresholds);

effectivePPCLog_permissive.absoluteAccuracy = nan(numGraphs, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_permissive.PCorrect = nan(numGraphs, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_permissive.LCA = nan(numGraphs, numRepetitions, maxN, numOptimalPerformanceThresholds);


data_extractedMIS = nan(numGraphs, numRepetitions, numCorrTresholds);
data_A_MIS = nan(numGraphs, numRepetitions);
data_NFeatures = nan(numGraphs, numRepetitions);
timing_findMIS = nan(numGraphs, numRepetitions);
timing_trainOnline = nan(numGraphs, numRepetitions);
timing_extractMIS = nan(numGraphs, numRepetitions);
timing_findOptimalControlPolicy = nan(numGraphs, numRepetitions, numOptimalPerformanceThresholds);
findOptimalControlPolicy_absoluteAccuracy = nan(numGraphs, numRepetitions);
findOptimalControlPolicy_PCorrect = nan(numGraphs, numRepetitions);
findOptimalControlPolicy_LCA = nan(numGraphs, numRepetitions);

NLog = nan(numGraphs, numRepetitions);

% data for particular graph specification
taskCardinalityLog_spec.absoluteAccuracy = nan(graphsPerSpecFolder, numRepetitions, numOptimalPerformanceThresholds);
taskCardinalityLog_spec.PCorrect = nan(graphsPerSpecFolder, numRepetitions, numOptimalPerformanceThresholds);
taskCardinalityLog_spec.LCA = nan(graphsPerSpecFolder, numRepetitions, numOptimalPerformanceThresholds);

maximumPerformanceCurveLog_spec.absoluteAccuracy = nan(graphsPerSpecFolder, numRepetitions, maxN);
maximumPerformanceCurveLog_spec.PCorrect = nan(graphsPerSpecFolder, numRepetitions, maxN);
maximumPerformanceCurveLog_spec.LCA = nan(graphsPerSpecFolder, numRepetitions, maxN);

minimumPerformanceCurveLog_spec.absoluteAccuracy = nan(graphsPerSpecFolder, numRepetitions, maxN);
minimumPerformanceCurveLog_spec.PCorrect = nan(graphsPerSpecFolder, numRepetitions, maxN);
minimumPerformanceCurveLog_spec.LCA = nan(graphsPerSpecFolder, numRepetitions, maxN);

effectivePPCLog_restrictive_spec.absoluteAccuracy = nan(graphsPerSpecFolder, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_restrictive_spec.PCorrect = nan(graphsPerSpecFolder, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_restrictive_spec.LCA = nan(graphsPerSpecFolder, numRepetitions, maxN, numOptimalPerformanceThresholds);

effectivePPCLog_permissive_spec.absoluteAccuracy = nan(graphsPerSpecFolder, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_permissive_spec.PCorrect = nan(graphsPerSpecFolder, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_permissive_spec.LCA = nan(graphsPerSpecFolder, numRepetitions, maxN, numOptimalPerformanceThresholds);


data_spec_extractedMIS = nan(graphsPerSpecFolder, numRepetitions, numCorrTresholds);
data_spec_A_MIS = nan(graphsPerSpecFolder, numRepetitions);
data_spec_NFeatures = nan(graphsPerSpecFolder, numRepetitions);
timing_spec_findMIS = nan(graphsPerSpecFolder, numRepetitions);
timing_spec_trainOnline = nan(graphsPerSpecFolder, numRepetitions);
timing_spec_extractMIS = nan(graphsPerSpecFolder, numRepetitions);
findOptimalControlPolicy_absoluteAccuracy_spec = nan(graphsPerSpecFolder, numRepetitions);
findOptimalControlPolicy_PCorrect_spec = nan(graphsPerSpecFolder, numRepetitions);
findOptimalControlPolicy_LCA_spec = nan(graphsPerSpecFolder, numRepetitions);

NLog_spec = nan(numGraphs, numRepetitions);

% accumulate all data
for A_ID = 1:numGraphs
    
    fieldtmp = graph_log(A_ID).taskCardinalityLog;
    if(isfield(fieldtmp, 'meanAcc'))
        taskCardinalityLog.absoluteAccuracy(A_ID,:,:) = graph_log(A_ID).taskCardinalityLog.meanAcc; 
        taskCardinalityLog.PCorrect(A_ID,:,:) = graph_log(A_ID).taskCardinalityLog.respProb; 
    else
        taskCardinalityLog.absoluteAccuracy(A_ID,:,:) = graph_log(A_ID).taskCardinalityLog.absoluteAccuracy; % meanAcc
        taskCardinalityLog.PCorrect(A_ID,:,:) = graph_log(A_ID).taskCardinalityLog.PCorrect; % respProb
    end
    taskCardinalityLog.LCA(A_ID,:,:) = graph_log(A_ID).taskCardinalityLog.LCA;
    
    maxCardinaility = size(graph_log(A_ID).maximumPerformanceCurveLog.absoluteAccuracy,2);
    maxUsedCardinality = size(graph_log(A_ID).effectivePPCLog_restrictive.absoluteAccuracy,2);
    
    maximumPerformanceCurveLog.absoluteAccuracy(A_ID,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.absoluteAccuracy;
    maximumPerformanceCurveLog.PCorrect(A_ID,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.PCorrect;
    maximumPerformanceCurveLog.LCA(A_ID,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.LCA;
    
    minimumPerformanceCurveLog.absoluteAccuracy(A_ID,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.absoluteAccuracy;
    minimumPerformanceCurveLog.PCorrect(A_ID,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.PCorrect;
    minimumPerformanceCurveLog.LCA(A_ID,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.LCA;
    
    effectivePPCLog_restrictive.absoluteAccuracy(A_ID,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_restrictive.absoluteAccuracy;
    effectivePPCLog_restrictive.PCorrect(A_ID,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_restrictive.PCorrect;
    effectivePPCLog_restrictive.LCA(A_ID,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_restrictive.LCA;
    
    effectivePPCLog_permissive.absoluteAccuracy(A_ID,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_permissive.absoluteAccuracy;
    effectivePPCLog_permissive.PCorrect(A_ID,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_permissive.PCorrect;
    effectivePPCLog_permissive.LCA(A_ID,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_permissive.LCA;
    
    
    if(ischar(graph_log(A_ID).extractedMIS))
        data_extractedMIS(A_ID,:,:) = str2num(graph_log(A_ID).extractedMIS');
    else
        data_extractedMIS(A_ID,:,:) = graph_log(A_ID).extractedMIS;
    end
    data_A_MIS(A_ID,:) = graph_log(A_ID).A_MIS;
    data_NFeatures(A_ID,:,:) = graph_log(A_ID).NFeatures;
    
    timing_findMIS(A_ID, :) = graph_log(A_ID).timing.findMIS;
    timing_trainOnline(A_ID, :) = graph_log(A_ID).timing.trainOnline;
    timing_extractMIS(A_ID, :) = graph_log(A_ID).timing.extractMIS;
    timing_findOptimalControlPolicy_absoluteAccuracy(A_ID,:) = graph_log(A_ID).timing.findOptimalControlPolicy_absoluteAccuracy;
    timing_findOptimalControlPolicy_PCorrect(A_ID,:) = graph_log(A_ID).timing.findOptimalControlPolicy_PCorrect;
    timing_findOptimalControlPolicy_LCA(A_ID,:) = graph_log(A_ID).timing.findOptimalControlPolicy_LCA;
    
    NLog(A_ID, :) = graph_log(A_ID).N;
end

% accumulate all data for particular configuration

spec_counter = 1;
for A_ID = 1:numGraphs
    if(strcmp(graph_log(A_ID).graphSpecsFolder, graphSpecsFolder_selected) && strcmp(graph_log(A_ID).samplingFolder, samplingFolder_selected))

        fieldtmp = graph_log(A_ID).taskCardinalityLog;
        if(isfield(fieldtmp, 'meanAcc'))
            taskCardinalityLog_spec.meanAcc(spec_counter,:,:) = graph_log(A_ID).taskCardinalityLog.meanAcc; 
            taskCardinalityLog_spec.PCorrect(spec_counter,:,:) = graph_log(A_ID).taskCardinalityLog.respProb; 
        else
            taskCardinalityLog_spec.meanAcc(spec_counter,:,:) = graph_log(A_ID).taskCardinalityLog.absoluteAccuracy; % meanAcc
            taskCardinalityLog_spec.PCorrect(spec_counter,:,:) = graph_log(A_ID).taskCardinalityLog.PCorrect; % respProb
        end
        taskCardinalityLog_spec.LCA(spec_counter,:,:) = graph_log(A_ID).taskCardinalityLog.LCA;

        maxCardinaility = size(graph_log(A_ID).maximumPerformanceCurveLog.absoluteAccuracy,2);
        maxUsedCardinality = size(graph_log(A_ID).effectivePPCLog_restrictive.absoluteAccuracy,2);
        
        maximumPerformanceCurveLog_spec.absoluteAccuracy(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.absoluteAccuracy;
        maximumPerformanceCurveLog_spec.PCorrect(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.PCorrect;
        maximumPerformanceCurveLog_spec.LCA(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.LCA;

        maximumPerformanceCurveLog_spec.absoluteAccuracy(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.absoluteAccuracy;
        maximumPerformanceCurveLog_spec.PCorrect(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.PCorrect;
        maximumPerformanceCurveLog_spec.LCA(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.LCA;

        effectivePPCLog_restrictive_spec.absoluteAccuracy(spec_counter,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_restrictive.absoluteAccuracy;
        effectivePPCLog_restrictive_spec.PCorrect(spec_counter,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_restrictive.PCorrect;
        effectivePPCLog_restrictive_spec.LCA(spec_counter,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_restrictive.LCA;
        
        effectivePPCLog_permissive_spec.absoluteAccuracy(spec_counter,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_permissive.absoluteAccuracy;
        effectivePPCLog_permissive_spec.PCorrect(spec_counter,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_permissive.PCorrect;
        effectivePPCLog_permissive_spec.LCA(spec_counter,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog_permissive.LCA;

        if(ischar(graph_log(A_ID).extractedMIS))
            data_spec_extractedMIS(spec_counter,:,:) = str2num(graph_log(A_ID).extractedMIS');
        else
            data_spec_extractedMIS(spec_counter,:,:) = graph_log(A_ID).extractedMIS;
        end
        data_spec_A_MIS(spec_counter,:) = graph_log(A_ID).A_MIS;
        data_spec_NFeatures(spec_counter,:,:) = graph_log(A_ID).NFeatures;

        timing_spec_findMIS(spec_counter, :) = graph_log(A_ID).timing.findMIS;
        timing_spec_trainOnline(spec_counter, :) = graph_log(A_ID).timing.trainOnline;
        timing_spec_extractMIS(spec_counter, :) = graph_log(A_ID).timing.extractMIS;
        timing_findOptimalControlPolicy_absoluteAccuracy_spec(spec_counter,:) = graph_log(A_ID).timing.findOptimalControlPolicy_absoluteAccuracy;
        timing_findOptimalControlPolicy_PCorrect_spec(spec_counter,:) = graph_log(A_ID).timing.findOptimalControlPolicy_PCorrect;
        timing_findOptimalControlPolicy_LCA_spec(spec_counter,:) = graph_log(A_ID).timing.findOptimalControlPolicy_LCA;

        NLog_spec(spec_counter, :) = graph_log(A_ID).N;
        
        spec_counter = spec_counter + 1;
    end
end

%% FIG 2a: Effective Parallel Processing Capacity as a function of N (for specific correlation threshold)

% parameters 
correlation_threshold = 0.8000;   % correlation threshold used to extract unweighted graph and compute MIS
optimalPerformanceThreshold = 0.90; % threshold on performance to decide whether the multitasking attempt was successful or not (only necessary for effective PPC metric)
performanceMetric = 'LCA';         % this could be absoluteAccuracy, PCorrect, LCA
divideByN = 0;
divideByTaskCardinality = 0;
showTitle = 1;
fontSize_gca = 14;

% find correlation threshold index 
corrThreshIdx = find(round(correlation_thresholds*1000) == round(correlation_threshold*1000),1);
if(isempty(corrThreshIdx))
    warning('Requested correlation threshold for MIS extraction does not exist in data set');
end

% find optimal performance threshold index
optPerformanceThreshIdx = find(round(goodPerformanceThresholds*100) == round(optimalPerformanceThreshold*100),1);
if(isempty(optPerformanceThreshIdx))
    warning('Requested optimal performance threshold does not exist in data set');
end

fig1 = figure(1);
set(fig1, 'Position', [500 500 700 300]);

% RESTRICTIVE REWARD REGIME
subplot(1,2,1);

% select corresponding performance
switch performanceMetric
    case 'absoluteAccuracy'
        
        performanceData = effectivePPCLog_restrictive.absoluteAccuracy;
        
    case 'PCorrect'
        
       performanceData = effectivePPCLog_restrictive.PCorrect;
       
    case 'LCA'
        
        performanceData = effectivePPCLog_restrictive.LCA;
        
end
         
% find all graphs with given degree and maximum N
maxN = max([graph_log(:).N]);

% initialize plot data
plotData= nan(maxN, maxN);
plotData_sem = nan(maxN, maxN);
x = 1:maxN;

% for each N
for N_idx = 1:maxN
    
    % find all networks with size N
    graphIDs = find([graph_log(:).N] == N_idx);
    
    % for task cardinalities
    for task_card = 1:maxN
        
        if(~divideByN)
            NDiv = 1;
        else
            NDiv = N_idx;
        end
        
        if(~divideByTaskCardinality)
            GammaDiv = 1;
        else
            GammaDiv = task_card;
        end
        
        plotData(N_idx, task_card) = nanmean(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx))) / GammaDiv / NDiv;
        plotData_sem(N_idx, task_card) = nanstd(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx) / GammaDiv / NDiv)) ...
                                                                / sqrt(sum(~isnan(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx)))));
    end
    
end

plotData = transpose(plotData);

% plot effective parallel processing capacity for each task set cardinality (gamma)
plotColors = colormap(jet(maxN));

% loop over task set cardinalities
for plotDataIdx = 1:size(plotData, 2)
    errorbar(x, plotData(:, plotDataIdx), plotData_sem(:, plotDataIdx), '-', 'LineWidth', lineWidth, 'color', plotColors(plotDataIdx,:)); hold on;
end

% legend
h = zeros(maxN, 1);
legendLabels = {};
for i = 1:maxN
    h(i) = plot(NaN, NaN, '-', 'LineWidth', lineWidth, 'color', plotColors(i,:));
    legendLabels{i} = ['N =' num2str(i)];
end
leg = legend(h, legendLabels,'Location','northeast');
set(leg, 'FontSize', fontSize_gca);

if(divideByN)
    ylim([0 0.4]);
else
    ylim([0 1]);
end
xlim([0 maxN+1]);
if(showTitle)
    title({'Restrictive', 'Reward Regime'}, 'FontSize', fontSize_title);
end
ylabel({'p(\theta = \gamma)' }, 'FontSize', fontSize_ylabel);
xlabel({'\gamma'}, 'FontSize', fontSize_ylabel);
set(gca, 'FontSize', fontSize_gca);

hold off;


% PERMISSIVE REWARD REGIME
subplot(1,2,2);

% select corresponding performance
switch performanceMetric
    case 'absoluteAccuracy'
        
        performanceData = effectivePPCLog_permissive.absoluteAccuracy;
        
    case 'PCorrect'
        
       performanceData = effectivePPCLog_permissive.PCorrect;
       
    case 'LCA'
        
        performanceData = effectivePPCLog_permissive.LCA;
        
end
         
% find all graphs with given degree and maximum N
maxN = max([graph_log(:).N]);

% initialize plot data
plotData= nan(maxN, maxN);
plotData_sem = nan(maxN, maxN);
x = 1:maxN;

% for each N
for N_idx = 1:maxN
    
    % find all networks with size N
    graphIDs = find([graph_log(:).N] == N_idx);
    
    % for task cardinalities
    for task_card = 1:maxN
        
        if(~divideByN)
            NDiv = 1;
        else
            NDiv = N_idx;
        end
        
        if(~divideByTaskCardinality)
            GammaDiv = 1;
        else
            GammaDiv = task_card;
        end
        
        plotData(N_idx, task_card) = nanmean(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx))) / GammaDiv / NDiv * task_card;
        plotData_sem(N_idx, task_card) = nanstd(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx) / GammaDiv / NDiv * task_card) ) ...
                                                                / sqrt(sum(~isnan(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx)))));
    end
    
end

plotData = transpose(plotData);

% plot effective parallel processing capacity for each task set cardinality (gamma)
plotColors = colormap(jet(maxN));

% loop over task set cardinalities
for plotDataIdx = 1:size(plotData, 2)
    errorbar(x, plotData(:, plotDataIdx), plotData_sem(:, plotDataIdx), '-', 'LineWidth', lineWidth, 'color', plotColors(plotDataIdx,:)); hold on;
end

% if(divideByN)
%     ylim([0 0.4]);
% else
%     ylim([0 1]);
% end

xlim([0 maxN+1]);
if(showTitle)
    title({'Permissive', 'Reward Regime'}, 'FontSize', fontSize_title);
end
ylabel({'\Theta_\gamma' }, 'FontSize', fontSize_ylabel);
xlabel('\gamma', 'FontSize', fontSize_ylabel);
set(gca, 'FontSize', fontSize_gca);

hold off;



























%% FIG 2b: Effective Parallel Processing Capacity as a function of N across all performance thresholds

% parameters 
correlation_threshold = 0.3000;   % correlation threshold used to extract unweighted graph and compute MIS
optimalPerformanceThreshold = 0.90; % threshold on performance to decide whether the multitasking attempt was successful or not (only necessary for effective PPC metric)
degree = 2;  % plotted cardinality relative to predicted MIS
performanceMetric = 'LCA';         % this could be absoluteAccuracy, PCorrect, LCA
divideByN = 1;
divideByTaskCardinality = 1;

% draw figure
fig1 = figure(1);
set(fig1, 'Position', [100 100 900 800]);

numRows = ceil(sqrt(length(correlation_thresholds)));
numCols = ceil(sqrt(length(correlation_thresholds)));

for threshIdx = 1:length(goodPerformanceThresholds)

    optimalPerformanceThreshold = goodPerformanceThresholds(threshIdx); 

    % find correlation threshold index 
    corrThreshIdx = find(round(correlation_thresholds*1000) == round(correlation_threshold*1000),1);
    if(isempty(corrThreshIdx))
        warning('Requested correlation threshold for MIS extraction does not exist in data set');
    end

    % find optimal performance threshold index
    optPerformanceThreshIdx = find(round(goodPerformanceThresholds*100) == round(optimalPerformanceThreshold*100),1);
    if(isempty(optPerformanceThreshIdx))
        warning('Requested optimal performance threshold does not exist in data set');
    end

    % select corresponding performance
    switch performanceMetric
        case 'absoluteAccuracy'

            performanceData = effectivePPCLog_restrictive.absoluteAccuracy;

        case 'PCorrect'

           performanceData = effectivePPCLog_restrictive.PCorrect;

        case 'LCA'

            performanceData = effectivePPCLog_restrictive.LCA;

    end

    % find all graphs with given degree and maximum N
    degreeIdx = find([graph_log(:).z] == degree);
    maxN = max([graph_log(degreeIdx).N]);

    % initialize plot data
    plotData= nan(maxN, maxN);
    plotData_sem = nan(maxN, maxN);
    x = 1:maxN;

    % for each N
    for N_idx = 1:maxN

        % find all networks with size N
        graphIDs = find([graph_log(:).z] == degree & [graph_log(:).N] == N_idx);

        % for task cardinalities
        for task_card = 1:maxN
            
            if(~divideByN)
                NDiv = 1;
            else
                NDiv = N_idx;
            end

            if(~divideByTaskCardinality)
                GammaDiv = 1;
            else
                GammaDiv = task_card;
            end

            plotData(N_idx, task_card) = nanmean(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx))) / GammaDiv / NDiv;
            plotData_sem(N_idx, task_card) = nanstd(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx)  / GammaDiv / NDiv)) ...
                                                                    / sqrt(sum(~isnan(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx)))));
        end

    end

    % plot effective parallel processing capacity for each task set cardinality (gamma)
    subplot(numRows, numCols, threshIdx);
    plotColors = colormap(jet(maxN));

    % loop over task set cardinalities
    for plotDataIdx = 1:size(plotData, 2)
        errorbar(x, plotData(:, plotDataIdx), plotData_sem(:, plotDataIdx), '-', 'LineWidth', lineWidth, 'color', plotColors(plotDataIdx,:)); hold on;
    end

    ylim([0 0.4]);
    xlim([3 maxN+1]);
    title(['thresh = ' num2str(optimalPerformanceThreshold)], 'FontSize', fontSize_title-1);
    ylabel({'Proportion Task Combinations', 'above Threshold / N' }, 'FontSize', fontSize_ylabel-2);
    xlabel('N', 'FontSize', fontSize_ylabel-2);
    set(gca, 'FontSize', fontSize_gca-2);

    hold off;

end


