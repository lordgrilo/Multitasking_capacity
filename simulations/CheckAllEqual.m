rng('default');
NPathways = 6;
NFeatures = 6;
samples = 1000;
sdScale = 0;
sameStimuliAcrossTasks = 1;
relevantTasks = [1     7    13    19     8    14    20    26     3     9    21    33     4    16    28    34     5    23    29    35    12    18    30]';

rng(0);
out1 = cell(9, 1);
[out1{1}, out1{2}, out1{3}, out1{4}, out1{5}, out1{6}, out1{7}, out1{8}, out1{9}] = createTaskPatterns(NPathways, NFeatures, samples, sdScale, sameStimuliAcrossTasks, relevantTasks);

rng(0);
out2 = cell(9, 1);
%[out2{1}, out2{2}, out2{3}, out2{4}, out2{5}, out2{6}, out2{7}, out2{8}, out2{9}] = createTaskPatterns_GenG(NPathways, NFeatures, samples, sdScale, sameStimuliAcrossTasks, relevantTasks);

for ii=1:9
   fprintf(1, '%d is %d\n', ii, isequal(out1{ii}, out2{ii})); 
end