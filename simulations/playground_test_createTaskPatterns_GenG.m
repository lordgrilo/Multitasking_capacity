clc;
sameStimuliAcrossTasks = 1;
samplesPerTask=  [];

[input, tasks, train, tasksIdxSgl, stimIdxSgl, inputSgl_mask, tasksSgl_mask, trainSgl_mask, multiCap, multiCap_con, multiCap_inc, relevantTasks, NPathways] = generateEnvironmentToGraph(A, NFeatures, samplesPerTask, sdScale, sameStimuliAcrossTasks);

figure(1);
subplot(1,3,1);
imagesc(input);

subplot(1,3,2);
imagesc(tasks);

subplot(1,3,3);
imagesc(train);


figure(2);
cap = 3;
subplot(1,3,1);
imagesc(multiCap{cap}.input);

subplot(1,3,2);
imagesc(multiCap{cap}.tasks);

subplot(1,3,3);
imagesc(multiCap{cap}.train);