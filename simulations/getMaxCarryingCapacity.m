function [pathwayCapacities, maxCarryingCapacity,  BK_MIS, A_bipartite, A_tasksIdx, A_dual] = getMaxCarryingCapacity(R_hidden, R_output, corr_threshold)
% calculates maximum carrying capacity of network based on hidden & output
% layer task representations. 
%
% INPUTS
% R_hidden: n x n similarity matrix of hidden layer task representations
% where n corresponds to the number of tasks
% R_output: n x n similarity matrix of output layer task representations
% where n corresponds to the number of tasks
%
% OUTPUTS
% pathwayCapacities: 3xn matrix, where n corresponds to the number of
% tasks. The first column indicates the hidden layer component and the
% second column the corresponding output layer component of the
% corresponding task. The third column indicates whether the task belongs
% to the maximum independent set and 0 if it belongs to the minimum
% independent set
%
% author: Sebastian Musslick
% this function uses code from Hasan K. Ozcimder & Biswadip Dey to
% calculate the adjacency matrix of the bipartite graph and code from
% Roberto Olmi to find the maximal independent set



% use similarity matricies to generate adjacency matrix of biparpite graph

% hidden layer
A = R_hidden;

currSet = [];

for i = 1:size(A,1)

    allSets(i).set = findFullSet(A, i, corr_threshold, currSet);
    currSet = [currSet allSets(i).set];
    
end
hiddenSets = allSets;

% output layer
A = R_output;

currSet = [];

for i = 1:size(A,1)

    allSets(i).set = findFullSet(A, i, corr_threshold, currSet);
    currSet = [currSet allSets(i).set];
    
end
outputSets = allSets;

% remove empty fields
hiddenSets(cellfun('isempty',{hiddenSets.set})) = [];
outputSets(cellfun('isempty',{outputSets.set})) = [];

hiddenComponents = length(hiddenSets);
outputComponents = length(outputSets);

%a = cellfun(@(x)sum(ismember([1 2],x)),{outputSets.set},'uni',false);

A_bipartite = zeros(hiddenComponents, outputComponents);
A_tasksIdx = zeros(hiddenComponents, outputComponents);

for row = 1:hiddenComponents
    
    links = cellfun(@(x)sum(ismember(hiddenSets(row).set,x)),{outputSets.set},'uni',false);
    A_bipartite(row,:) = [links{:}];
    
end

% inefficient for now
for row = 1:hiddenComponents
   
    for hiddenTaskRep = 1:length(hiddenSets(row).set)
        
        taskIdx = hiddenSets(row).set(hiddenTaskRep);
        
       for col = 1:outputComponents
           if(ismember(taskIdx, outputSets(col).set))
               A_tasksIdx(row, col) = taskIdx;
               break;
           end
       end
    end
 
end

% compute MIS from bipartite graph
[maxCarryingCapacity, A_dual, pathwayCapacities,  BK_MIS] = getMISFromBipartiteGraph(A_bipartite);

end