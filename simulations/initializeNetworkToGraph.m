function [network, fixToBasisSet_log] = initializeNetworkToGraph(A, network, varargin)
% initializeNetworkToGraph(A, NNModel)
% 
% A ...adjacency matrix of bipartite graph (A)
% rows of A refer to input nodes
% columns of A refer to output nodes
%
% network ...instance of NNmodel
%
% initializeNetworkToGraph(A, NNModel, fixInputHiddenWeights, fixTaskHiddenWeights)
%
% optional boolean inputs fixInputHiddenWeights and fixTaskHiddenWeights
% specify if weights from input layer to hidden layer or task layer to
% hidden layer shall be fixed. E.g. for initializeNetworkToGraph(A, NNModel, 0, 1)
% the network will initialize and fix only weights from the task to the
% hidden layer while weights from the input to the hidden layer will be
% randmly initialized and can be modified through learning
%
%% preliminary steps

% parse input arguments
if(~isempty(varargin))
    fixInputHiddenWeights = varargin{1};
else
    fixInputHiddenWeights = 1;
end

if(length(varargin) >= 2)
    fixTaskHiddenWeights = varargin{2};
else
    fixTaskHiddenWeights = 1;
end

if(length(varargin) >= 3)
    fixToBasisSet = varargin{3};
else
    fixToBasisSet = 1;
end

nLayers = length(network.Nhidden);

% determine number of input & output components
[inputComponents, outputComponents] = size(A);

NPathways = max(size(A));

% set network bias
network.bias_weight = -5;
network.hidden_bias = network.bias_weight;
network.output_bias = network.bias_weight;

% maximum activation value of a hidden unit
maxActivationValueByTask = 0.1;
maxInputValue = 1;

% calculate required weight magnitude to activate a hidden unit from a
% single input unit
% w_IH = (- log(1/maxActivationValue - 1) - network.bias_weight) / maxInputValue;
w_IH = 2;

% calculate required weight from task units to activate hidden unit that
% receives input from stimulus
w_TH = (- log(1/maxActivationValueByTask - 1) - network.bias_weight) / maxInputValue;

% quick check if we have the right amount of input units
NFeatures = network.Ninput / inputComponents;
if(round(NFeatures) ~= NFeatures)
    error('Number of input units does not match required number of input components (row number of A).');
end

% check if we do have enoug task units
if(network.Ntask < NPathways^2)
    error(['Network does not have enough task units for specified number of input and output dimensions. At least ' num2str(NPathways^2) ' task units are required.']);
end

% check if we do have enoug hidden units
if(fixToBasisSet == 1)
    if(network.Nhidden < NFeatures * inputComponents)
        error(['Network does not have enough hidden units for specified number of input components and features per input dimension. At least ' num2str(NFeatures*inputComponents) ' hidden units are required.']);
    end
else
    if(network.Nhidden < NFeatures * inputComponents * outputComponents)
        error(['Network does not have enough hidden units for specified number of input components and features per input dimension. At least ' num2str(NFeatures*inputComponents) ' hidden units are required.']);
    end    
end

%% determine input-hidden weights

if(fixInputHiddenWeights)

    % set input weights
    subWeights = eye(NFeatures * inputComponents, NFeatures * inputComponents) * w_IH;
    
    if(~fixToBasisSet) % if fixing to tensor product, make copies of each input dimension in hidden layer
        subWeights_org = subWeights;
        for o = 1:(outputComponents-1)
            subWeights = [subWeights; subWeights_org];
        end
    end

    % overwrite network weights
    network.weights.W_IH = zeros(network.Nhidden,network.Ninput);
    network.weights.W_IH(1:size(subWeights,1), 1:size(subWeights,2)) = subWeights;

    % fix weights
    network.setFixedWeights([network.getFixedWeights()  network.W_INPUT_HIDDEN]);

end

%% determine task-hidden weights
fixToBasisSet_log = [];
integer = round(clock* 100);
s = RandStream('mt19937ar','Seed',integer(end));
RandStream.setGlobalStream(s);

if(fixTaskHiddenWeights)

    for layer = 1:nLayers
        
        if(fixToBasisSet == 2)
            sample = rand;
            sample
            if(sample < 0.5)
                fixToBasisSet_log(layer) = 1;
            else
                fixToBasisSet_log(layer) = 0;
            end
        else
            fixToBasisSet_log(layer) = fixToBasisSet;
        end
        
        if(nLayers == 1)
            network.weights.W_TH = zeros(network.Nhidden, network.Ntask);
        else
            network.weights.W_TH{layer} = zeros(network.Nhidden(1), network.Ntask);
        end
    end

    for hiddenCIdx = 1:inputComponents

        for outputCIdx = 1:outputComponents

            % determine which hidden units a task projects to based on common
            % hidden (input) component
            hiddenTaskRep_basis = ((hiddenCIdx-1)*NFeatures + 1) :  ((hiddenCIdx-1)*NFeatures + NFeatures);
            hiddenTaskRep_tensor = (outputCIdx-1) * NFeatures*inputComponents + (((hiddenCIdx-1)*NFeatures + 1) :  ((hiddenCIdx-1)*NFeatures + NFeatures));

            % for every edge (task) in the bipartie graph, create a task
            % pathway
            if(A(hiddenCIdx, outputCIdx) > 0)

                % determine task ID
                task = (hiddenCIdx-1) * NPathways + outputCIdx;

                for layer = 1:nLayers
                    
                    if(nLayers == 1)
                        % adjust weights for task
                        if(fixToBasisSet_log(layer) == 1)
                            network.weights.W_TH(hiddenTaskRep_basis, task) = w_TH;
                        else
                            network.weights.W_TH(hiddenTaskRep_tensor, task) = w_TH;
                        end
                    else
                        if(fixToBasisSet_log(layer) == 1)
                            network.weights.W_TH{layer}(hiddenTaskRep_basis, task) = w_TH;
                        else
                            network.weights.W_TH{layer}(hiddenTaskRep_tensor, task) = w_TH;
                        end
                    end
                end

            end

        end

    end

    % fix weights
    network.setFixedWeights([network.getFixedWeights() network.W_TASK_HIDDEN]);

end

end
