function Giovanni_Sim1_V8_fun_plots_offline(plotID)
%%
% openfig('MSE_300h_plot.fig','new','visible')
%
% clear all;
plotID = str2num(plotID);
clc;

% dataset specification
datafilePrefix = 'Giovanni_Simulation_V8_h500';
correlation_threshold = 0.35;
optimalPerformanceThreshold = 0.90;
graphSpecsFolder_selected = 'N_6_z_1';
samplingFolder_selected = 'ER'; % CM ER regular
graphsPerSpecFolder = 40;
maxN = 6;

logFolder = 'logfiles/';

% load data
files = dir(logFolder);

% get valid file names
validFileNames = {};
for i =1:length(files)
    % check if this is a desired data file
    if(~isempty(strfind(files(i).name, datafilePrefix)) && isempty(strfind(files(i).name, '0.2')))
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
        graph_log(fileNameIdx).effectivePPCLog = effectivePPCLog;
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

effectivePPCLog.absoluteAccuracy = nan(numGraphs, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog.PCorrect = nan(numGraphs, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog.LCA = nan(numGraphs, numRepetitions, maxN, numOptimalPerformanceThresholds);

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

effectivePPCLog_spec.absoluteAccuracy = nan(graphsPerSpecFolder, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_spec.PCorrect = nan(graphsPerSpecFolder, numRepetitions, maxN, numOptimalPerformanceThresholds);
effectivePPCLog_spec.LCA = nan(graphsPerSpecFolder, numRepetitions, maxN, numOptimalPerformanceThresholds);

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
    maxUsedCardinality = size(graph_log(A_ID).effectivePPCLog.absoluteAccuracy,2);
    
    maximumPerformanceCurveLog.absoluteAccuracy(A_ID,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.absoluteAccuracy;
    maximumPerformanceCurveLog.PCorrect(A_ID,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.PCorrect;
    maximumPerformanceCurveLog.LCA(A_ID,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.LCA;
    
    minimumPerformanceCurveLog.absoluteAccuracy(A_ID,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.absoluteAccuracy;
    minimumPerformanceCurveLog.PCorrect(A_ID,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.PCorrect;
    minimumPerformanceCurveLog.LCA(A_ID,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.LCA;
    
    effectivePPCLog.absoluteAccuracy(A_ID,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog.absoluteAccuracy;
    effectivePPCLog.PCorrect(A_ID,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog.PCorrect;
    effectivePPCLog.LCA(A_ID,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog.LCA;
    
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
        maxUsedCardinality = size(graph_log(A_ID).effectivePPCLog.absoluteAccuracy,2);
        
        maximumPerformanceCurveLog_spec.absoluteAccuracy(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.absoluteAccuracy;
        maximumPerformanceCurveLog_spec.PCorrect(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.PCorrect;
        maximumPerformanceCurveLog_spec.LCA(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).maximumPerformanceCurveLog.LCA;

        maximumPerformanceCurveLog_spec.absoluteAccuracy(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.absoluteAccuracy;
        maximumPerformanceCurveLog_spec.PCorrect(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.PCorrect;
        maximumPerformanceCurveLog_spec.LCA(spec_counter,:,1:maxCardinaility) = graph_log(A_ID).minimumPerformanceCurveLog.LCA;

        effectivePPCLog_spec.absoluteAccuracy(spec_counter,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog.absoluteAccuracy;
        effectivePPCLog_spec.PCorrect(spec_counter,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog.PCorrect;
        effectivePPCLog_spec.LCA(spec_counter,:,1:maxUsedCardinality,:) = graph_log(A_ID).effectivePPCLog.LCA;

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

%% FIG 1a: Performance as a function of predicted MIS (for specific correlation threshold)
if(plotID == 1)

% parameters 
correlation_threshold = 0.3000;   % correlation threshold used to extract unweighted graph and compute MIS
optimalPerformanceThreshold = 0.90; % threshold on performance to decide whether the multitasking attempt was successful or not (only necessary for effective PPC metric)
cardinality_window = -3:3;  % plotted cardinality relative to predicted MIS
performanceMetric = 'LCA';         % this could be absoluteAccuracy, PCorrect, LCA
performanceType = 'max';                                % this could be min, max, effective
showSingleGraph = 0; % 200

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
    
    case 'MSE'
        
        switch performanceType
            case 'max'
                performanceData = (1-maximumPerformanceCurveLog.absoluteAccuracy).^2;
            case 'min'
                performanceData = (1-minimumPerformanceCurveLog.absoluteAccuracy).^2;
            case 'effective'
                performanceData = (1-(effectivePPCLog.absoluteAccuracy(:,:,:,optPerformanceThreshIdx))).^2;
        end
    
    case 'absoluteAccuracy'
        
        switch performanceType
            case 'max'
                performanceData = maximumPerformanceCurveLog.absoluteAccuracy;
            case 'min'
                performanceData = minimumPerformanceCurveLog.absoluteAccuracy;
            case 'effective'
                performanceData = (effectivePPCLog.absoluteAccuracy(:,:,:,optPerformanceThreshIdx));
        end
        
    case 'PCorrect'
        
        switch performanceType
            case 'max'
                performanceData = maximumPerformanceCurveLog.PCorrect;
            case 'min'
                performanceData = minimumPerformanceCurveLog.PCorrect;
            case 'effective'
                performanceData = (effectivePPCLog.PCorrect(:,:,:,optPerformanceThreshIdx));
        end
        
    case 'LCA'
        
        switch performanceType
            case 'max'
                performanceData = maximumPerformanceCurveLog.LCA;
            case 'min'
                performanceData = minimumPerformanceCurveLog.LCA;
            case 'effective'
                performanceData = (effectivePPCLog.LCA(:,:,:,optPerformanceThreshIdx));
        end
        
end

% first transform data such that performance curve is aligned with predicted N
accuracyCurve = nan(numGraphs, length(cardinality_window));

performanceData = squeeze(nanmean(performanceData, 2)); % take mean performance over all repetitions per graph
MISData = squeeze(mode(data_extractedMIS, 2));  % take mode of MIS across all repetitions per graph
MISData_specific = squeeze(MISData(:, corrThreshIdx)); % pick MIS corresponding to selected correlation threshold
middleIndex = find(cardinality_window == 0);

for graphIdx = 1:size(performanceData, 1)
    
    minIndexOrg = max(1, MISData_specific(graphIdx) + cardinality_window(1));
    maxIndexOrg = min(maxN, MISData_specific(graphIdx) + cardinality_window(end));
    
    minIndexNormalized = middleIndex - min(MISData_specific(graphIdx), middleIndex) + 1;
    maxIndexNormalized = middleIndex + min(maxN - MISData_specific(graphIdx),  middleIndex-1);
    
    accuracyCurve(graphIdx, minIndexNormalized:maxIndexNormalized) = performanceData(graphIdx,minIndexOrg:maxIndexOrg);
    
end

% plot all performance curves
fig1 = figure(1);
set(fig1, 'visible','off');
set(fig1, 'Position', [500 500 550 250]);
plotColors = colormap(autumn(maxN));
plotColors = plotColors(fliplr(1:size(plotColors,1)),:);


colorOffset = 0.2;
for row = 1:size(plotColors, 1)
    for col = 1:size(plotColors, 2)
        plotColors(row, col) = min(plotColors(row, col) + colorOffset, 1);
    end
end

if(strcmp(performanceMetric, 'MSE'))
    scalar = 1;
else
    scalar = 100;
end

if(any(MISData_specific == 0))
    warning('Found MIS of 0. Fixing to 1.');
    MISData_specific(MISData_specific == 0) = 1;
end

if(showSingleGraph == 0)
    for graphIdx = 1:size(accuracyCurve, 1)
        plot(accuracyCurve(graphIdx,:) * scalar, '-', 'LineWidth', lineWidth-2, 'color', plotColors(MISData_specific(graphIdx),:)); hold on;
    end
else
    graphIdx = showSingleGraph;
    plot(accuracyCurve(graphIdx,:) * scalar, '-', 'LineWidth', lineWidth-2, 'color', 'k'); hold on;
end

% plot line to indicate MIS
plot([middleIndex middleIndex], [0 100], '--k', 'LineWidth', lineWidth);

% legend
if(showSingleGraph == 0 | showSingleGraph ~= 0)
    h = zeros(maxN, 1);
    legendLabels = {};
    for i = 1:maxN
        h(i) = plot(NaN, NaN, '-', 'LineWidth', lineWidth, 'color', plotColors(i,:)); 
        legendLabels{i} = ['MIS =' num2str(i)];
    end
    legendLabels{end+1} = '';
    leg = legend(h, legendLabels,'Location','eastoutside');
    set(leg, 'FontSize', fontSize_gca);
end

if(showSingleGraph == 0)
    set(leg, 'TextColor', 'k');
    set(leg, 'Color', 'w');
end

% axes
xTickLabels = {};
if(showSingleGraph == 0)
    for i = 1:length(cardinality_window)
        if(sign(cardinality_window(i)) > 0)
            signLabel = ['+' num2str(cardinality_window(i))];
        elseif(sign(cardinality_window(i)) < 0)
            signLabel = num2str(cardinality_window(i));
        else
            signLabel = '';
        end
        xTickLabels{i} = ['MIS' signLabel];
    end
else
    for i = 1:length(cardinality_window)
        xTickLabels{i} = num2str(MISData_specific(graphIdx) + min(cardinality_window) + i -1);
    end
end


set(gca, 'XTick', 1:length(cardinality_window));
set(gca, 'XTickLabel', xTickLabels);
if(strcmp(performanceMetric, 'LCA'))
    ylabelString = {'Multitasking Accuracy in %', '(LCA)'};
    ylim([0 100]);
    ylabel(ylabelString, 'FontSize', fontSize_ylabel);
elseif(strcmp(performanceMetric, 'absoluteAccuracy'))
    ylabelString = {'Multitasking Accuracy in %', '(Absolute Accuracy)'};
    ylim([0 0.08]);
    ylabel(ylabelString, 'FontSize', fontSize_ylabel);
elseif(strcmp(performanceMetric, 'PCorrect'))
    ylabelString = {'Multitasking Accuracy in %', '(Luce Ratio)'};
    ylabel(ylabelString, 'FontSize', fontSize_ylabel);
end
xlim([0.5, length(cardinality_window) + 0.5]);
% title([performanceMetric ' ' performanceType], 'FontSize', fontSize_title);
xlabel('Task Set Size', 'FontSize', fontSize_ylabel);
set(gca, 'FontSize', fontSize_gca);

hold off;

set(gcf, 'Color', 'w');
set(gca, 'Color', 'w');
set(gca, 'xColor', 'k');
set(gca, 'yColor', 'k');
set(gca, 'zColor', 'k');
set(leg, 'TextColor', 'k');
set(leg, 'Color', 'w');

saveas(fig1,[datafilePrefix 'MSE_plot'],'fig')

end
%% perform statistical test

% select only graphs with 1 data point before and 1 data point after MIS
testCurve = accuracyCurve;
removeIdx = [];
middle = round(size(accuracyCurve,2)/2);
for graphIdx = 1:size(accuracyCurve, 1)
    if(any(isnan(accuracyCurve(graphIdx, [middle-1 middle middle+1]))))
        removeIdx = [removeIdx graphIdx];
    end
end
testCurve(removeIdx,:) = [];

% compute bias of fitted sigmoid for each curve
testBiases = nan(1, size(testCurve,1));
x_full = (1:size(testCurve,2))-middle;
for graphIdx = 1:size(testCurve, 1)
    
    nanIdx = isnan(testCurve(graphIdx,:));
    x = x_full;
    y = testCurve(graphIdx,:);
    x(nanIdx) = [];
    y(nanIdx) = [];
    
    performanceMax = max(y);
    performanceMin = min(y);
    
    [param] = sigm_fit(x,y, [performanceMin, performanceMax , NaN , NaN], [performanceMin performanceMax 0 -2]);
    testBiases(graphIdx) = param(3);
    
end

% perform two one-sided t-tests
[H,P,CI, STATS] = ttest(testBiases, 0, 'tail', 'right');
disp(['t-test bias above 0, t(' num2str(STATS.df) ') = ' num2str(STATS.tstat) ', p = ' num2str(P)]);
[H,P,CI, STATS] = ttest(testBiases, 1, 'tail', 'left');
disp(['t-test bias below 1, t(' num2str(STATS.df) ') = ' num2str(STATS.tstat) ', p = ' num2str(P)]);
    

%% FIG 1b: Performance as a function of predicted MIS across all correlation thresholds
if(plotID == 1)

plottedPerformanceMetrics{1} = 'LCA';
plottedPerformanceMetrics{2} = 'PCorrect';
plottedPerformanceMetrics{3} = 'absoluteAccuracy';

for metricIdx = 1:length(plottedPerformanceMetrics)

% parameters 
optimalPerformanceThreshold = 0.90; % threshold on performance to decide whether the multitasking attempt was successful or not (only necessary for effective PPC metric)
cardinality_window = -3:3;  % plotted cardinality relative to predicted MIS
performanceMetric = plottedPerformanceMetrics{metricIdx};         % this could be absoluteAccuracy, PCorrect, LCA
performanceType = 'max';                                % this could be min, max, effective

plotted_correlation_thresholds = [0.1:0.1:0.9 0.95];

% draw figure
fig1 = figure(1);
set(fig1, 'Position', [100 100 1600 500]);

numRows = ceil(sqrt(length(plotted_correlation_thresholds)));
numCols = ceil(sqrt(length(plotted_correlation_thresholds)));
numRows = 2;
numCols = 5;

for threshIdx = 1:length(plotted_correlation_thresholds)

    correlation_threshold = plotted_correlation_thresholds(threshIdx); 

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

    % select corresponding performance
    switch performanceMetric

        case 'absoluteAccuracy'

            ylabelString = {'Multitasking Accuracy in %', '(Absolute Accuracy)'};
            switch performanceType
                case 'max'
                    performanceData = maximumPerformanceCurveLog.absoluteAccuracy;
                case 'min'
                    performanceData = minimumPerformanceCurveLog.absoluteAccuracy;
                case 'effective'
                    performanceData = squeeze(effectivePPCLog.absoluteAccuracy(:,:,:,optPerformanceThreshIdx));
            end

        case 'PCorrect'

            ylabelString = {'Multitasking Accuracy in %', '(Luce Ratio)'};
            switch performanceType
                case 'max'
                    performanceData = maximumPerformanceCurveLog.PCorrect;
                case 'min'
                    performanceData = minimumPerformanceCurveLog.PCorrect;
                case 'effective'
                    performanceData = squeeze(effectivePPCLog.PCorrect(:,:,:,optPerformanceThreshIdx));
            end

        case 'LCA'

            ylabelString = {'Multitasking Accuracy in %', '(LCA)'};
            switch performanceType
                case 'max'
                    performanceData = maximumPerformanceCurveLog.LCA;
                case 'min'
                    performanceData = minimumPerformanceCurveLog.LCA;
                case 'effective'
                    performanceData = squeeze(effectivePPCLog.LCA(:,:,:,optPerformanceThreshIdx));
            end

    end

    % first transform data such that performance curve is aligned with predicted N
    accuracyCurve = nan(numGraphs, length(cardinality_window));

    performanceData = squeeze(nanmean(performanceData, 2)); % take mean performance over all repetitions per graph
    MISData = squeeze(mode(data_extractedMIS, 2));  % take mode of MIS across all repetitions per graph
    MISData_specific = squeeze(MISData(:, corrThreshIdx)); % pick MIS corresponding to selected correlation threshold
    middleIndex = find(cardinality_window == 0);

    for graphIdx = 1:size(performanceData, 1)

        minIndexOrg = max(1, MISData_specific(graphIdx) + cardinality_window(1));
        maxIndexOrg = min(maxN, MISData_specific(graphIdx) + cardinality_window(end));

        minIndexNormalized = middleIndex - min(MISData_specific(graphIdx), middleIndex) + 1;
        maxIndexNormalized = middleIndex + min(maxN - MISData_specific(graphIdx),  middleIndex-1);

        accuracyCurve(graphIdx, minIndexNormalized:maxIndexNormalized) = performanceData(graphIdx,minIndexOrg:maxIndexOrg);

    end
    
    if(any(MISData_specific == 0))
        warning('Found MIS of 0. Fixing to 1.');
        MISData_specific(MISData_specific == 0) = 1;
    end

    % plot all performance curves
    subplot(numRows, numCols, threshIdx);
    
    plotColors = colormap(autumn(maxN));
    plotColors = plotColors(fliplr(1:size(plotColors,1)),:);


    for graphIdx = 1:size(accuracyCurve, 1)
        plot(accuracyCurve(graphIdx,:) * 100, '-', 'LineWidth', lineWidth-2, 'color', plotColors(MISData_specific(graphIdx),:)); hold on;
    end

    % axes
    xTickLabels = {};
    for i = 1:length(cardinality_window)
        if(sign(cardinality_window(i)) > 0)
            signLabel = ['+' num2str(cardinality_window(i))];
        elseif(sign(cardinality_window(i)) < 0)
            signLabel = num2str(cardinality_window(i));
        else
            signLabel = '';
        end
        xTickLabels{i} = ['MIS' signLabel];
    end

    set(gca, 'XTick', 1:length(cardinality_window));
    set(gca, 'XTickLabel', xTickLabels);
    ylim([0 100]);
    xlim([0.5, length(cardinality_window) + 0.5]);
    set(gca, 'FontSize', fontSize_gca-6);
    ylabel(ylabelString, 'FontSize', fontSize_ylabel-2);
    xlabel('Task Set Size', 'FontSize', fontSize_ylabel-2);
    title(['\theta = ' num2str(correlation_threshold)], 'FontSize', fontSize_title-1);

    % plot line to indicate MIS
    plot([middleIndex middleIndex], [0 100], '--k', 'LineWidth', lineWidth);

    hold off;

end

saveas(fig1,[datafilePrefix 'MSE_CorrThresh_' plottedPerformanceMetrics{metricIdx} '_plot'],'fig')

end

end
%% FIG 1c: Performance as a function of predicted MIS across all network sizes
if(plotID == 1)
    
plottedPerformanceMetrics{1} = 'LCA';
plottedPerformanceMetrics{2} = 'PCorrect';
plottedPerformanceMetrics{3} = 'absoluteAccuracy';

for metricIdx = 1:length(plottedPerformanceMetrics)

% parameters 
correlation_threshold = 0.3000;   % correlation threshold used to extract unweighted graph and compute MIS
optimalPerformanceThreshold = 0.90; % threshold on performance to decide whether the multitasking attempt was successful or not (only necessary for effective PPC metric)
cardinality_window = -3:3;  % plotted cardinality relative to predicted MIS
performanceMetric = plottedPerformanceMetrics{metricIdx};         % this could be absoluteAccuracy, PCorrect, LCA
performanceType = 'max';                                % this could be min, max, effective
MIS_type = 'true';         % extracted, true

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

% draw figure
fig1 = figure(1);
set(fig1, 'Position', [100 100 1000 400]);

% select corresponding performance
switch performanceMetric

    case 'absoluteAccuracy'

        ylabelString = {'Multitasking Accuracy', 'in % (Absolute Accuracy)'};
        switch performanceType
            case 'max'
                performanceData = maximumPerformanceCurveLog.absoluteAccuracy;
            case 'min'
                performanceData = minimumPerformanceCurveLog.absoluteAccuracy;
            case 'effective'
                performanceData = squeeze(effectivePPCLog.absoluteAccuracy(:,:,:,optPerformanceThreshIdx));
        end

    case 'PCorrect'

        ylabelString = {'Multitasking Accuracy', 'in % (Luce Ratio)'};
        switch performanceType
            case 'max'
                performanceData = maximumPerformanceCurveLog.PCorrect;
            case 'min'
                performanceData = minimumPerformanceCurveLog.PCorrect;
            case 'effective'
                performanceData = squeeze(effectivePPCLog.PCorrect(:,:,:,optPerformanceThreshIdx));
        end

    case 'LCA'

        ylabelString = {'Multitasking Accuracy', 'in % (LCA)'};
        switch performanceType
            case 'max'
                performanceData = maximumPerformanceCurveLog.LCA;
            case 'min'
                performanceData = minimumPerformanceCurveLog.LCA;
            case 'effective'
                performanceData = squeeze(effectivePPCLog.LCA(:,:,:,optPerformanceThreshIdx));
        end

end

% first transform data such that performance curve is aligned with predicted N
accuracyCurve = nan(numGraphs, length(cardinality_window));

performanceData = squeeze(nanmean(performanceData, 2)); % take mean performance over all repetitions per graph
if(strcmp(MIS_type, 'extracted'))
    MISData = squeeze(mode(data_extractedMIS, 2));  % take mode of MIS across all repetitions per graph
    MISData_specific = squeeze(MISData(:, corrThreshIdx)); % pick MIS corresponding to selected correlation threshold
elseif(strcmp(MIS_type, 'true'))
    MISData_specific = data_A_MIS;
else
    error('MIS type doesn''t exist. Use either ''extracted'' or ''true''.');
end
middleIndex = find(cardinality_window == 0);

for graphIdx = 1:size(performanceData, 1)

    if(~isnan(MISData_specific(graphIdx)))
    minIndexOrg = max(1, MISData_specific(graphIdx) + cardinality_window(1));
    maxIndexOrg = min(maxN, MISData_specific(graphIdx) + cardinality_window(end));

    minIndexNormalized = middleIndex - min(MISData_specific(graphIdx), middleIndex) + 1;
    maxIndexNormalized = middleIndex + min(maxN - MISData_specific(graphIdx),  middleIndex-1);

    accuracyCurve(graphIdx, minIndexNormalized:maxIndexNormalized) = performanceData(graphIdx,minIndexOrg:maxIndexOrg);
    end
    
end

largestN = max([graph_log(:).N]);

numRows = ceil(sqrt(largestN));
numCols = ceil(sqrt(largestN));

if(any(MISData_specific == 0))
        warning('Found MIS of 0. Fixing to 1.');
        MISData_specific(MISData_specific == 0) = 1;
end

plottedNetworkSizes = 4:9;
numRows = 2;
numCols = 3;
    
for N_idx = 1:length(plottedNetworkSizes)
    
    current_N = plottedNetworkSizes(N_idx);
    
    % plot all performance curves
    subplot(numRows, numCols, N_idx);
    
    plotColors = colormap(autumn(maxN));
    plotColors = plotColors(fliplr(1:size(plotColors,1)),:);

    
    graphsOfSizeN = find([graph_log(:).N] == current_N);

    for graphIdx = 1:length(graphsOfSizeN);
        
        currentGraphIdx = graphsOfSizeN(graphIdx);
        if(~isnan(MISData_specific(currentGraphIdx)))
            plot(accuracyCurve(currentGraphIdx,:) * 100, '-', 'LineWidth', lineWidth-2, 'color', plotColors(MISData_specific(currentGraphIdx),:)); hold on;
        end
    end

    % axes
    xTickLabels = {};
    for i = 1:length(cardinality_window)
        if(sign(cardinality_window(i)) > 0)
            signLabel = ['+' num2str(cardinality_window(i))];
        elseif(sign(cardinality_window(i)) < 0)
            signLabel = num2str(cardinality_window(i));
        else
            signLabel = '';
        end
        xTickLabels{i} = ['MIS' signLabel];
    end

    set(gca, 'XTick', 1:length(cardinality_window));
    set(gca, 'XTickLabel', xTickLabels);
    ylim([0 100]);
    xlim([0.5, length(cardinality_window) + 0.5]);
    set(gca, 'FontSize', fontSize_gca-6);
    ylabel(ylabelString, 'FontSize', fontSize_ylabel-2);
    xlabel('Task Set Size', 'FontSize', fontSize_ylabel-2);
    title(['D = ' num2str(current_N)], 'FontSize', fontSize_title-1);

    % plot line to indicate MIS
    plot([middleIndex middleIndex], [0 100], '--k', 'LineWidth', lineWidth);

    hold off;

end

saveas(fig1,[datafilePrefix 'MSE_N_' plottedPerformanceMetrics{metricIdx} '_plot'],'fig')

end

end
%% FIG 2a: Effective Parallel Processing Capacity as a function of N (for specific correlation threshold)
if(plotID == 4)
    
% parameters 
correlation_threshold = 0.8000;   % correlation threshold used to extract unweighted graph and compute MIS
optimalPerformanceThreshold = 0.90; % threshold on performance to decide whether the multitasking attempt was successful or not (only necessary for effective PPC metric)
degree = 2;  % plotted cardinality relative to predicted MIS
performanceMetric = 'PCorrect';         % this could be absoluteAccuracy, PCorrect, LCA
divideByN = 0;
divideByTaskCardinality = 1;

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
        
        performanceData = effectivePPCLog.absoluteAccuracy;
        
    case 'PCorrect'
        
       performanceData = effectivePPCLog.PCorrect;
       
    case 'LCA'
        
        performanceData = effectivePPCLog.LCA;
        
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
        plotData_sem(N_idx, task_card) = nanstd(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx) / GammaDiv / NDiv)) ...
                                                                / sqrt(sum(~isnan(squeeze(performanceData(graphIDs, :, task_card, optPerformanceThreshIdx)))));
    end
    
end

% plot effective parallel processing capacity for each task set cardinality (gamma)
fig1 = figure(1);
set(fig1, 'Position', [500 500 380 300]);
plotColors = colormap(autumn(maxN));
plotColors = plotColors(fliplr(1:size(plotColors,1)),:);


% loop over task set cardinalities
for plotDataIdx = 1:size(plotData, 2)
    errorbar(x, plotData(:, plotDataIdx), plotData_sem(:, plotDataIdx), '-', 'LineWidth', lineWidth, 'color', plotColors(plotDataIdx,:)); hold on;
end

% legend
h = zeros(maxN, 1);
legendLabels = {};
for i = 1:maxN
    h(i) = plot(NaN, NaN, '-', 'LineWidth', lineWidth, 'color', plotColors(i,:));
    legendLabels{i} = ['\gamma =' num2str(i)];
end
leg = legend(h, legendLabels,'Location','eastoutside');
set(leg, 'FontSize', fontSize_gca);

if(divideByN)
    ylim([0 0.4]);
else
    ylim([0 1]);
end
xlim([3 maxN+1]);
title([performanceMetric], 'FontSize', fontSize_title);
ylabel({'Proportion Of Sucessfully Performed', 'Tasks In The Set \gamma (divided by N)' }, 'FontSize', fontSize_ylabel);
xlabel('Network Size N', 'FontSize', fontSize_ylabel);
set(gca, 'FontSize', fontSize_gca);

hold off;

end
%% FIG 2b: Effective Parallel Processing Capacity as a function of N across all performance thresholds
if(plotID == 5)
    
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

            performanceData = effectivePPCLog.absoluteAccuracy;

        case 'PCorrect'

           performanceData = effectivePPCLog.PCorrect;

        case 'LCA'

            performanceData = effectivePPCLog.LCA;

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
    plotColors = colormap(autumn(maxN));
    plotColors = plotColors(fliplr(1:size(plotColors,1)),:);


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


end
%% FIG 3: MIS Of Imposed Bipartite Graph Vs. Optimal Signal Policy vs. Extracted MIS (averaged across repetititons)
if(plotID == 6)
fontSize = 14;

MISLog = nanmean(data_spec_A_MIS,2);
taskCardinalityLog = squeeze(nanmean(data_spec_optimalControlPolicy_meanAcc(:,:,optPerformanceThreshIdx), 2));
extractedMISLog = squeeze(nanmean(data_spec_extractedMIS,2));

y = transpose([MISLog taskCardinalityLog extractedMISLog]);

fig1 = figure(1);
set(fig1, 'Position', [100 100 1200 600]);
imagesc(y);
caxis([1 max(unique(MISLog))]);
hcb = colorbar('XTick', 1:max(unique(MISLog)));
title(hcb,'MIS')

set(gca, 'YTick', [1:size(y, 1)]);
set(gca, 'XTick', [1:size(y, 2)]);
ylabels = {'MIS Of Imposed Task Weight Structure', 'Optimal Control Policy', ['Extracted MIS With Correlation Threshold = ' num2str(correlation_thresholds(1))]};

for i = 2:length(correlation_thresholds)
    ylabels{end + 1} = ['= ' num2str(correlation_thresholds(i))];
end
set(gca, 'YTickLabels', ylabels);
set(gca, 'fontSize', fontSize);
xlabel(['Samples Of ' num2str(NPathways) ' x ' num2str(NPathways) ' Bipartite Graph' ], 'fontSize', fontSize);
title(['Optimal Performance Threshold = ' num2str(goodPerformanceThresholds(optPerformanceThreshIdx))]);
set(gca,'xgrid', 'off', 'ygrid', 'on', 'gridlinestyle', '-', 'xcolor', 'k', 'ycolor', 'k');
set(fig1, 'Color', 'w');

end
