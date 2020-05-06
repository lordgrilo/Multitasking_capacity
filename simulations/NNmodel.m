%% description
% this class implements a neural network model with two input layers, one
% hidden layer and an output layer. One input layer represents the current
% stimulus, the other input layer represents the current task. The input
% layer projects to the hidden layer and the hidden layer to the output
% layer. The task input projects to both the hidden layer and the output
% layer. Learning is implemented using backpropagation with optional weight
% penalization and optional weight constraint over the weights coming from
% the task input layer.
%
% author: Sebastian Musslick

%%

classdef NNmodel < handle
    
    properties(SetAccess = public)
        trainSet;           % training data
        inputSet;           % input data
        taskSet;            % task data
        
        Ninput;             % number of input units
        Ntask;              % number of task units
        Nhidden;            % number of hidden units
        Noutput;            % number of output units
        
        hiddenPathSize      % size of one path (group of units) in hidden layer
        outputPathSize      % size of one path (group of units) in output layer
        NPathways           % optional parameter: number of processing pathways (can be used to infer input and output dimensions)
        
        bias_weight;        % default bias weight
        hidden_bias;        % hidden bias weight
        output_bias;        % output bias weight
        
        coeff;              % learning rate
        thresh;             % stopping criterion
        decay;              % weight penalization parameter
        weights;            % network weights
        fixedWeights;    % indicates which weights are fixed
       
        init_scale;         % scales for initialized weights
        init_task_scale     % scales for initialized weights from task to hidden layer
        
        hidden_act;         % current hidden activity
        output_act;         % current output activity
        
        MSE_log;            % full MSE curve
        MSE_patterns_log    % full MSE curve for all patterns
        CE_log;                 % full cross entropy error curve
        CE_patterns_log;    % full cross entropy error curve for all patterns
        CF_log;                 % full classification error curve
        CF_patterns_log;    % full classification error curve for all patterns
        DimCF_log;          % full dimension-wise classification error curve
        DimCF_patterns_log; % full dimension-wise classification error curve for all patterns
        
        hidden_log;         % full hidden activity for input set
        output_log;         % full output for input set
               
        silence;            % Should we print diagnostic messages
    end
    
    properties (SetAccess = protected)
        isFixedInputHiddenW % Logical value indicating whether weights are fixed between input and hidden 
        isFixedTaskHiddenW  % Logical value indicating whether weights are fixed between task and hidden
        isFixedTaskOutputW  % Logical value indicating whether weights are fixed between task and output
        isFixedHiddenOutputW% Logical value indicating whether weights are fixed between output and hidden
        isFixedBiasHiddenW  % Logical value indicating whether weights are fixed between bias and hidden
        isFixedBiasOutputW  % Logical value indicating whether weights are fixed between bias and output
    end
    
    properties (Constant)
        
        W_INPUT_HIDDEN = 1;
        W_TASK_HIDDEN = 2;
        W_HIDDEN_OUTPUT = 3;
        W_TASK_OUTPUT = 4;
        W_BIAS_HIDDEN= 5;
        W_BIAS_OUTPUT = 6;
        
    end
    
    methods
        
        % constructor
        function this = NNmodel(varargin)
            
            % make a copy of existing network object
            if(isa(varargin{1},'NNmodel'))
                copy = varargin{1};
                this.trainSet       = copy.trainSet;
                this.inputSet       = copy.inputSet;
                this.taskSet        = copy.taskSet;
                this.Ninput         = copy.Ninput;
                this.Ntask          = copy.Ntask;
                this.Nhidden        = copy.Nhidden;
                this.Noutput        = copy.Noutput;
                this.bias_weight    = copy.bias_weight;
                this.hidden_bias    = copy.hidden_bias;
                this.output_bias    = copy.output_bias;
                this.coeff          = copy.coeff;
                this.weights        = copy.weights;
                this.setFixedWeights([]);
                this.init_scale     = copy.init_scale;
                this.init_task_scale        = copy.init_task_scale;
                this.hidden_act     = copy.hidden_act;
                this.output_act     = copy.output_act;
                this.MSE_log        = copy.MSE_log;
                this.CE_log        = copy.CE_log;
                this.CF_log        = copy.CF_log;
                this.MSE_patterns_log        = copy.MSE_patterns_log;
                this.hidden_log     = copy.hidden_log;
                this.output_log     = copy.output_log;
                this.hiddenPathSize = copy.hiddenPathSize;
                this.outputPathSize = copy.outputPathSize;
                this.thresh         = copy.thresh;
                this.decay          = copy.decay;
                this.NPathways          = copy.NPathways;
                this.silence = copy.silence;
                return;
                
                % parse arguments
            else
               % number of hidden layer units
               this.Nhidden = varargin{1};  
            end
            
            % learning rate
            if(length(varargin)>=2)
               this.coeff = varargin{2};  
            else
               this.coeff = 0.3; 
            end
            
            % weight from bias units to hidden and output units
            if(length(varargin)>=3)
               this.bias_weight = varargin{3};  
            else
               this.bias_weight = -1; 
            end
            
            % maximum absolute magnitude of initial weights
            if(length(varargin)>=4)
               this.init_scale = varargin{4};  
            else
               this.init_scale = 1; 
            end
            
            % mean-squared error stopping criterion for learning
            if(length(varargin)>=5)
               this.thresh = varargin{5};  
            else
               this.thresh = 0.01; 
            end
            
            % weight penalization parameter
            if(length(varargin)>=6)
               this.decay = varargin{6};  
            else
               this.decay = 0.02; 
            end
            
            % size of one path (group of units) in hidden layer
            if(length(varargin)>=7)
               this.hiddenPathSize = varargin{7};  
            else
               this.hiddenPathSize = 1; 
            end
            
            % size of one path (group of units) in output layer
            if(length(varargin)>=8)
               this.outputPathSize = varargin{8};  
            else
               this.outputPathSize = 1; 
            end
            
            % initialization noise from task to hidden layer
            this.init_task_scale = this.init_scale;
            
            % assign optional parameters
            this.NPathways = [];
            
            % no fixed weights by default
            this.setFixedWeights([]);
            
            % By default, don't silence output
            this.silence = false;

        end
        
        function my_printf(this, varargin)  
            if(~this.silence)
                fprintf(1, varargin{:});
            end
        end
        
        function setFixedWeights(this, fixedWeights)
            
            this.isFixedInputHiddenW = false;
            this.isFixedTaskHiddenW = false;
            this.isFixedTaskOutputW = false;
            this.isFixedHiddenOutputW = false;
            
            if(ismember(this.W_INPUT_HIDDEN, fixedWeights))
                this.isFixedInputHiddenW = true;
                this.fixedWeights.W_IH = true;
            end
            
            if(ismember(this.W_TASK_HIDDEN, fixedWeights))
                this.isFixedTaskHiddenW = true;
                this.fixedWeights.W_TH = true(1, length(this.Nhidden));
            end
            
            if(ismember(this.W_HIDDEN_OUTPUT, fixedWeights))
                this.isFixedHiddenOutputW = true;
                this.fixedWeights.W_HO = true;
            end
            
            if(ismember(this.W_TASK_OUTPUT, fixedWeights))
                this.isFixedTaskOutputW = true;
                this.fixedWeights.W_TO = true;
            end
            
            if(ismember(this.W_BIAS_HIDDEN, fixedWeights))
                this.isFixedBiasHiddenW = true;
                this.fixedWeights.W_BH = true;
            end
            
            if(ismember(this.W_BIAS_OUTPUT, fixedWeights))
                this.isFixedBiasOutputW = true;
                this.fixedWeights.W_BO = true;
            end
            
        end
        
        function fixedWeights = getFixedWeights(this)
           fixedWeights = [];
      
           if(this.isFixedInputHiddenW) fixedWeights = [fixedWeights, this.W_INPUT_HIDDEN]; end
           if(this.isFixedTaskHiddenW) fixedWeights = [fixedWeights, this.W_TASK_HIDDEN]; end
           if(this.isFixedHiddenOutputW) fixedWeights = [fixedWeights, this.W_HIDDEN_OUTPUT]; end
           if(this.isFixedTaskOutputW) fixedWeights = [fixedWeights, this.W_TASK_OUTPUT]; end
           if(this.isFixedBiasHiddenW) fixedWeights = [fixedWeights, this.W_BIAS_HIDDEN]; end
           if(this.isFixedBiasOutputW) fixedWeights = [fixedWeights, this.W_BIAS_OUTPUT]; end
        end
        
        % configure net: set up weights and network size depending on
        % trainig patterns
        function configure(this, varargin)
            
            % evaluate input arguments
            if(length(varargin)==1)
                
                % use configuration of existing net
                if(isa(varargin{1},'NNmodel'))
                   netObj = varargin{1};
                   this.weights = netObj.weights;
                   this.Ninput = netObj.Ninput;
                   this.Ntask = netObj.Ntask;
                   this.Noutput = netObj.Noutput;
                   this.Nhidden = netObj.Nhidden;
                   this.hidden_bias = netObj.hidden_bias;
                   this.output_bias = netObj.output_bias;
                   this.bias_weight = netObj.bias_weight;
                   this.hiddenPathSize = netObj.hiddenPathSize;
                   this.outputPathSize = netObj.outputPathSize;
                   this.thresh         = netObj.thresh;
                   this.decay          = netObj.decay;
                end
            else
                
                % set input patterns if provided by arguments
                if(length(varargin)>=3)
                   this.inputSet = varargin{1};
                   this.taskSet =  varargin{2};
                   this.trainSet = varargin{3};
                   
                
                % check if network has inputs, tasks and output patterns 
                else
                   if(isempty(this.inputSet) || isempty(this.taskSet) || isempty(this.trainSet))
                       error('Input set and training set need to be specified in order to configure network.');
                   end  
                end
                
                % set number of units for each layer
                this.Ninput = size(this.inputSet,2);
                this.Ntask = size(this.taskSet,2);
                this.Noutput = size(this.trainSet,2);
                if(isempty(this.Nhidden))
                    this.Nhidden = size(this.inputSet,2);
                end
                
                % set bias inputs for hidden & output layers
                if(isempty(this.bias_weight))
                    this.bias_weight = -1;           
                end
                this.hidden_bias = repmat(this.bias_weight,1,1);    % bias is the same for all hidden units
                this.output_bias = repmat(this.bias_weight,1,1);    % bias is the same for all output units 
                            
                % weight initialization (random using seed)
                %rand('state',sum(100*clock));
                
                % set up weight matrices
                this.weights.W_IH = (-1 +2.*rand(this.Nhidden(1),this.Ninput))*this.init_scale;      % input-to-hidden weights
                
                if(length(this.Nhidden) > 1)
                    for hiddenLayer = 1:length(this.Nhidden)
                        if(length(this.init_task_scale) < hiddenLayer)
                                if(length(this.init_task_scale) == 0)
                                    this.weights.W_TH{hiddenLayer} =  zeros(this.Nhidden(hiddenLayer),this.Ntask);
                                    this.fixedWeights.W_TH(hiddenLayer) = 1;
                                else
                                    this.weights.W_TH{hiddenLayer} =  (-1 +2.*rand(this.Nhidden(hiddenLayer),this.Ntask))*this.init_task_scale;
                                end
                        else
                                % standard initialization by weight range
                                 this.weights.W_TH{hiddenLayer} = (-1 +2.*rand(this.Nhidden(hiddenLayer),this.Ntask))*this.init_task_scale(hiddenLayer);
                        end
                        if(hiddenLayer > 1)
                            this.weights.W_HH{hiddenLayer-1} = (-1 +2.*rand(this.Nhidden(hiddenLayer),this.Nhidden(hiddenLayer-1)))*this.init_scale;
                        end

                        this.weights.W_BH{hiddenLayer} = ones(this.Nhidden(hiddenLayer),1);                                         % bias-to-hidden weights
                    end
                else
                    if(isempty(this.init_task_scale))
                        this.init_task_scale = this.init_scale;
                    end
                    this.weights.W_TH = (-1 +2.*rand(this.Nhidden,this.Ntask))*this.init_task_scale;
                    this.weights.W_BH = ones(this.Nhidden,1);                                         % bias-to-hidden weights
                end
                
                this.weights.W_TO = (-1 +2.*rand(this.Noutput,this.Ntask))*this.init_scale;              
                this.weights.W_HO = (-1 +2.*rand(this.Noutput,this.Nhidden(1)))*this.init_scale;     % output-to-hidden weights
                this.weights.W_BO = ones(this.Noutput,1);                                         % bias-to-output weights
            end
            
            this.setFixedWeights(this.W_BIAS_HIDDEN);
            this.setFixedWeights(this.W_BIAS_OUTPUT);
            this.fixedWeights.W_IH = false;
            this.fixedWeights.W_TH = false;
            this.fixedWeights.W_TO = false;
            this.fixedWeights.W_HO = false;
            this.fixedWeights.W_HH = false(1, length(this.Nhidden));
        end
        
        % train the network on all patterns
        function [] = trainOnline(this, iterations, varargin)
            
            % parse arguments: input patterns, task pattenrs, output patterns
            if(length(varargin)>=2)
               inputData =  varargin{1};
               taskData = varargin{2};
               trainData =  varargin{3};
            else
               inputData = this.inputSet;
               taskData = this.taskSet;
               trainData = this.trainSet;
            end
            
            % check if input and task datasets have equal number of patterns (rows)
            if(size(inputData,1) ~= size(taskData,1))
                error('Task data has to have same number of rows as input data.');
            end
            
            % check if input and training datasets have equal number of patterns (rows)
            if(size(inputData,1) ~= size(trainData,1))
                error('Training data has to have same number of rows as input data.');
            end
            
            Ndatasets = size(inputData,1);              % total number of datasets
            this.MSE_log = zeros(1,iterations);         % log mean-squared error (MSE)
            this.CE_log = zeros(1,iterations);         % log cross-entropy error
            this.CF_log = zeros(1,iterations);         % log classification error
            this.DimCF_log = zeros(1,iterations);         % log dimension classification error
            this.MSE_patterns_log = zeros(Ndatasets, iterations);  % log MSE for all patterns
            this.CE_patterns_log = zeros(Ndatasets, iterations);  % log cross-entropy error for all patterns
            this.CF_patterns_log = zeros(Ndatasets, iterations);  % log classification error for all patterns
            this.DimCF_patterns_log = zeros(Ndatasets, iterations);  % log dimension classification error for all patterns
            
            % for each learning iteration
            for i = 1:iterations
               
               % randomize training set for each learning iteration
               order = randperm(size(inputData,1));
               inputData = inputData(order,:);
               taskData = taskData(order,:);
               trainData = trainData(order,:);
                
               MSE = zeros(1,Ndatasets);                            % current mean-squared error for all patterns (datasets)
               this.hidden_log = zeros(Ndatasets,this.Nhidden(1));     % current log activity of hidden units for each dataset
               this.output_log = zeros(Ndatasets,this.Noutput);     % current log activity of output units for each dataset
               
               % loop through all the patterns (online training)
               for dset = 1:Ndatasets
                  [MSE(dset)] = trainTrial(this, inputData(dset,:),  taskData(dset,:), trainData(dset,:)); % trains weights on current pattern
                  this.hidden_log(dset,:) = this.hidden_act{end}';       % log hidden unit activity for this pattern
                  this.output_log(dset,:) = this.output_act';       % log output unit activity for this pattern
               end
               
               [this.MSE_log(i),  this.MSE_patterns_log(:,i)] = this.calculateMeanSquaredError(this.output_log, trainData);
               [this.CE_log(i), this.CE_patterns_log(:,i)] = this.calculateCrossEntropyError(this.output_log, trainData);
               [this.CF_log(:,i), this.CF_patterns_log(:,i)] = this.calculateClassificationError(this.output_log, trainData);
               [this.DimCF_log(i), this.DimCF_patterns_log(:,i)] = this.calculateDimClassificationError(this.output_log, trainData);
               
               % stop learning if the mean-squared error reaches a certain
               % threshold
               if(this.MSE_log(i)) < this.thresh
                  break; 
               end
               
               this.my_printf('iteration:%d\n', i);
               
            end
            
        end
        
        % train a trial
        function [MSE] = trainTrial(this, input, task, train)
            
            
               % simulate trial, retrieve activation values for hidden and
               % output layer
               this.runTrial(input, task);
               
               % weight update (backpropagation):
               % delta_w = -coeff * delta * x_i
               % delta_w      ...weight adjustment
               % coeff        ...learning rate
               % delta        ...represents backpropagated error
               % x_i          ...activation of sending unit

               % calculate delta's for output layer: delta_output = (output_act - train) * f_act'(netj)
               % delta_output ...backpropagated error for output units
               % output_act   ...output activity
               % train        ...correct output
               % f_act'(netj) ...first derivative of activation function of output units with respect to the net input
               error_term = (this.output_act - transpose(train));
               error_term(isnan(error_term)) = 0;                   % if target value is not specified (NaN), then error should be 0
               delta_output = error_term .* this.output_act .* (1 - this.output_act);

               % calculate delta's for hidden layer: delta_hidden = sum(delta_output * W_HO) * f_act'(netj)
               % delta_hidden ...backpropagated error for hidden units
               % delta_output ...backpropagated error for output units
               % W_HO         ...weights from hidden (columns) to output layer (rows)
               % f_act'(netj) ...first derivative of activation function of hidden units with respect to the net input
               for hiddenLayerIdx = 1:length(this.Nhidden)
                   
                   hiddenLayer = length(this.Nhidden)-hiddenLayerIdx+1;

                   if(length(this.Nhidden) > 1)
                       % hidden layer that connects to output layer
                       if(hiddenLayer == length(this.Nhidden))
                            delta_hidden{hiddenLayer} = sum(repmat(delta_output,1,size(this.weights.W_HO,2)) .* this.weights.W_HO,1)' .* this.hidden_act{hiddenLayer} .* (1 - this.hidden_act{hiddenLayer});
                       else
                       % preceding hidden layers
                            delta_hidden{hiddenLayer} = sum(repmat(delta_hidden{hiddenLayer+1} ,1,size(this.weights.W_HH{hiddenLayer},2)) .* this.weights.W_HH{hiddenLayer},1)' .* this.hidden_act{hiddenLayer} .* (1 - this.hidden_act{hiddenLayer});
                       end
                   else
                       delta_hidden = sum(repmat(delta_output,1,size(this.weights.W_HO,2)) .* this.weights.W_HO,1)' .* this.hidden_act .* (1 - this.hidden_act);
                   end
                   
               end
               
               % if a pathway size for the hidden unit layer is specified, 
               % then the deltas for groups of hidden layer units that 
               % receive input from the task input layer will be averaged. 
               % The path size specifies the number of hidden units in a
               % group. Each task unit projects to all groups of hidden
               % units with the constraint that the projecting weights will 
               % be the same for the units within a group
               delta_hiddenTask = delta_hidden;
               for hiddenLayer = 1:length(this.Nhidden)
                   if(this.hiddenPathSize(min(hiddenLayer, length(this.hiddenPathSize))) > 1) % no need to do averaging if hidden pathway size is 1
                       % average paths for hidden-to-task backprop
                       Npaths = this.Nhidden(hiddenLayer)/this.hiddenPathSize(hiddenLayer);
                       refVec = repmat(1:Npaths,this.hiddenPathSize(hiddenLayer),1);
                       refVec = refVec(:);
                
                       if(length(this.Nhidden) > 1)
                           for i = 1:Npaths
                               delta_hiddenTask{hiddenLayer}(refVec==i) = mean(delta_hidden{hiddenLayer}(refVec==i));
                           end
                       else
                               delta_hiddenTask(refVec==i) = mean(delta_hidden(refVec==i));
                       end
                   end 
               end
               
               % if a pathway size for the output unit layer is specified, 
               % then the deltas for groups of output layer units that 
               % receive input from the task input layer will be averaged. 
               % The path size specifies the number of output units in a
               % group. Each task unit projects to all groups of output
               % units with the constraint that the projecting weights will 
               % be the same for the units within a group
               delta_outputTask = delta_output;
               if(this.outputPathSize > 1)
                   % average paths for output-to-task backprop
                   Npaths = this.Noutput/this.outputPathSize;
                   refVec = repmat(1:Npaths,this.outputPathSize,1);
                   refVec = refVec(:);
                   for i = 1:Npaths
                       delta_outputTask(refVec==i) = mean(delta_output(refVec==i));
                   end
               end

               % adjust weights from input to hidden units
               if(~this.fixedWeights.W_IH)
                    if(length(this.Nhidden) > 1)
                        this.weights.W_IH = this.weights.W_IH - (this.coeff * delta_hidden{1} * input) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_IH));
                        weightLog.dW_IH = - this.coeff * delta_hidden{1} * input - this.coeff * this.decay * sign(this.weights.W_IH) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_IH));
                    else
                        this.weights.W_IH = this.weights.W_IH - (this.coeff * delta_hidden * input) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_IH));
                        weightLog.dW_IH = - this.coeff * delta_hidden * input - this.coeff * this.decay * sign(this.weights.W_IH) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_IH));
                    end
               else
                   weightLog.dW_IH = 0;
               end
               
               % adjust weights to hidden layers
               for hiddenLayer = 1:length(this.Nhidden)
                   
                   % adjust weights from hidden to hidden layer
                   if(hiddenLayer < length(this.Nhidden))
                       if(~this.fixedWeights.W_HH(hiddenLayer))
                            this.weights.W_HH{hiddenLayer} = this.weights.W_HH{hiddenLayer} - (this.coeff * delta_hidden{hiddenLayer+1} * this.hidden_act{hiddenLayer}') - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_HH{hiddenLayer}));
                            weightLog.dW_HH{hiddenLayer} = - (this.coeff * delta_hidden{hiddenLayer+1} * this.hidden_act{hiddenLayer}') - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_HH{hiddenLayer}));
                       else
                            weightLog.dW_HH{hiddenLayer} = 0;
                       end
                   end
               
                   % adjust weights from task to hidden units
                   if(length(this.Nhidden) > 1)
                       if(~this.fixedWeights.W_TH(hiddenLayer))
                            this.weights.W_TH{hiddenLayer} = this.weights.W_TH{hiddenLayer} - (this.coeff * delta_hiddenTask{hiddenLayer} * task .* this.dg(this.weights.W_TH{hiddenLayer})) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_TH{hiddenLayer}));
                            weightLog.dW_TH{hiddenLayer} = - this.coeff * delta_hiddenTask{hiddenLayer} * task * this.dg(this.weights.W_TH{hiddenLayer}) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_TH{hiddenLayer}));
                       else
                            weightLog.dW_TH{hiddenLayer} = 0;
                       end

                       % turn bias learning off 
%                        if(~this.fixedWeights.W_BH(hiddenLayer))
%                             this.weights.W_BH{hiddenLayer} = this.weights.W_BH{hiddenLayer} - this.coeff * delta_hidden{hiddenLayer} * this.hidden_bias - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_BH{hiddenLayer})); 
%                        end
                   else
                       if(~this.fixedWeights.W_TH)
                            this.weights.W_TH = this.weights.W_TH - (this.coeff * delta_hiddenTask * task .* this.dg(this.weights.W_TH)) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_TH));
                            weightLog.dW_TH = - this.coeff * delta_hiddenTask * task * this.dg(this.weights.W_TH) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_TH));
                       else
                            weightLog.dW_TH = 0;
                       end

                       % turn bias learning off 
%                        if(~this.fixedWeights.W_BH(hiddenLayer))
%                             this.weights.W_BH = this.weights.W_BH - this.coeff * delta_hidden * this.hidden_bias - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_BH)); 
%                        end
                   end
               
               end
               
               % adjust weights from task to output units
               if(~this.fixedWeights.W_TO)
                   this.weights.W_TO = this.weights.W_TO - (this.coeff * delta_outputTask * task) .* this.dg(this.weights.W_TO) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_TO));
                   weightLog.dW_TO = - (this.coeff * delta_outputTask * task) .* this.dg(this.weights.W_TO) - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_TO));
               else
                   weightLog.dW_TO = 0;
               end
               
               % adjust weights from hidden to output units
               if(~this.fixedWeights.W_HO)
                   if(length(this.Nhidden) > 1)
                       this.weights.W_HO = this.weights.W_HO - (this.coeff * delta_output * this.hidden_act{end}') - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_HO));
                       weightLog.dW_HO = - (this.coeff * delta_output * this.hidden_act{end}') - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_HO));
                   else
                       this.weights.W_HO = this.weights.W_HO - (this.coeff * delta_output * this.hidden_act') - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_HO));
                        weightLog.dW_HO = - (this.coeff * delta_output * this.hidden_act') - (this.coeff * this.decay / size(this.inputSet,1) * sign(this.weights.W_HO));
                   end
               else
                   weightLog.dW_HO = 0;
               end
               
               % calculate mean-squared error
               [~,  MSE] = this.calculateMeanSquaredError(this.output_act', train);

        end
        
         % train the network on all patterns
        function [] = trainBatch(this, iterations, varargin)
            
            if(length(this.Nhidden) > 1) 
                error('Batch training not implemented for deep nets.');
            end
            
            % parse arguments: input patterns, task pattenrs, output patterns
            if(length(varargin)>=2)
               inputData =  varargin{1};
               taskData = varargin{2};
               trainData =  varargin{3};
            else
               inputData = this.inputSet;
               taskData = this.taskSet;
               trainData = this.trainSet;
            end
            
            % check if input and task datasets have equal number of patterns (rows)
            if(size(inputData,1) ~= size(taskData,1))
                error('Task data has to have same number of rows as input data.');
            end
            
            % check if input and training datasets have equal number of patterns (rows)
            if(size(inputData,1) ~= size(trainData,1))
                error('Training data has to have same number of rows as input data.');
            end
            
            Ndatasets = size(inputData,1);              % total number of datasets
            this.MSE_log = zeros(1,iterations);         % log mean-squared error (MSE)
            this.MSE_patterns_log = zeros(Ndatasets, iterations); % log MSE for all patterns
            
            % calculate groups of hidden units, if a pathway size from task to hidden layer
            % is specified (for details, see below)
            if(this.hiddenPathSize > 1) % no need to do averaging if hidden pathway size is 1
               % average paths for hidden-to-task backprop
               Npaths_hidden = this.Nhidden/this.hiddenPathSize;
               refVec_hidden = repmat(1:Npaths_hidden,this.hiddenPathSize,1);
               refVec_hidden = refVec_hidden(:);
            end
            
            % calculate groups of output units, if a pathway size from task to output layer
            % is specified (for details, see below)
            if(this.outputPathSize > 1) % no need to do averaging if output pathway size is 1
               % average paths for output-to-task backprop
               Npaths_output = this.Noutput/this.outputPathSize;
               refVec_output = repmat(1:Npaths_output,this.outputPathSize,1);
               refVec_output = refVec_output(:);
            end
            
            
            % for each learning iteration
            for i = 1:iterations
                
               this.hidden_log = zeros(Ndatasets,this.Nhidden);     % current log activity of hidden units for each dataset
               this.output_log = zeros(Ndatasets,this.Noutput);     % current log activity of output units for each dataset
               
               % simulate trial, retrieve activation values for hidden and
               % output layer for each dataset
               [outData, hiddenData, MSE] = this.runSet(inputData, taskData, trainData);
               
               % weight update (backpropagation):
               % delta_w = -coeff * delta * x_i
               % delta_w      ...weight adjustment
               % coeff        ...learning rate
               % delta        ...represents backpropagated error
               % x_i          ...activation of sending unit

               % calculate delta's for output layer: delta_output = (output_act - train) * f_act'(netj)
               % delta_output ...backpropagated error for output units
               % output_act   ...output activity
               % train        ...correct output
               % f_act'(netj) ...first derivative of activation function of output units with respect to the net input
               error_term = transpose(outData - trainData);
               error_term(isnan(error_term)) = 0;                   % if target value is not specified (NaN), then error should be 0
               delta_output = error_term .* this.output_act .* (1 - this.output_act);
               
               % calculate delta's for hidden layer: delta_hidden = sum(delta_output * W_HO) * f_act'(netj)
               % delta_hidden ...backpropagated error for hidden units
               % delta_output ...backpropagated error for output units
               % W_HO         ...weights from hidden (columns) to output layer (rows)
               
               % f_act'(netj) ...first derivative of activation function of hidden units with respect to the net input
               % delta_hidden = sum(repmat(delta_output,1,size(this.weights.W_HO,2)) .* this.weights.W_HO,1)' .* this.hidden_act .* (1 - this.hidden_act);
               delta_hidden = delta_output' * this.weights.W_HO .* hiddenData .* (1 - hiddenData);
               
               % if a pathway size for the hidden unit layer is specified, 
               % then the deltas for groups of hidden layer units that 
               % receive input from the task input layer will be averaged. 
               % The path size specifies the number of hidden units in a
               % group. Each task unit projects to all groups of hidden
               % units with the constraint that the projecting weights will 
               % be the same for the units within a group
               delta_hiddenTask = delta_hidden;
               if(this.hiddenPathSize > 1) % no need to do averaging if hidden pathway size is 1
                   % average paths for hidden-to-task backprop
                   [GroupId,~,index_j]=unique(refVec_hidden);
                   GroupMean=arrayfun(@(k) mean(delta_hidden(:,index_j==k),2),1:length(GroupId), 'UniformOutput', 0);
                   delta_hiddenTask=[GroupMean{index_j}];
               end
               
               % if a pathway size for the output unit layer is specified, 
               % then the deltas for groups of output layer units that 
               % receive input from the task input layer will be averaged. 
               % The path size specifies the number of output units in a
               % group. Each task unit projects to all groups of output
               % units with the constraint that the projecting weights will 
               % be the same for the units within a group
               delta_output = delta_output';
               delta_outputTask = delta_output;
               if(this.outputPathSize > 1) % no need to do averaging if hidden pathway size is 1
                   % average paths for hidden-to-task backprop
                   [GroupId,~,index_j]=unique(refVec_output);
                   GroupMean=arrayfun(@(k) mean(delta_output(:,index_j==k),2),1:length(GroupId), 'UniformOutput', 0);
                   delta_outputTask=[GroupMean{index_j}];
               end
               
               % adjust weights from input to hidden units
               if(~this.isFixedInputHiddenW)
                    this.weights.W_IH = this.weights.W_IH - this.coeff * delta_hidden' * inputData - Ndatasets * this.coeff * this.decay * sign(this.weights.W_IH);
               end
               % adjust weights from task to hidden units
               if(~this.isFixedTaskHiddenW)
                    this.weights.W_TH = this.weights.W_TH - this.coeff * delta_hiddenTask' * taskData .* 1;%this.dg(this.weights.W_TH);
               end

               % adjust weights from task to output units
               if(~this.isFixedTaskOutputW)
                    this.weights.W_TO = this.weights.W_TO - this.coeff * delta_outputTask' * taskData .* 1;%this.dg(this.weights.W_TO);
               end
               % adjust weights from hidden to output units
               if(~this.isFixedHiddenOutputW)
                    this.weights.W_HO = this.weights.W_HO - this.coeff * delta_output' * hiddenData - this.coeff * this.decay * sign(this.weights.W_HO);
               end
               
               % this.decay * sign(this.weights.W_IH) ...penalize weights
               % this.dg(this.weights.W_TO)           ...derivative of transformation function of the weights
               
               % learning of bias weights is turned off for now
               % this.weights.W_BH = this.weights.W_BH - this.coeff * sum(delta_hidden,2) * this.hidden_bias; 
               % this.weights.W_BO = this.weights.W_BO - this.coeff * sum(delta_output,2) * this.output_bias;

               % calculate mean-squared error
               MSE = sum((outData - trainData).^2,2);
               
               [this.MSE_log(i),  this.MSE_patterns_log(:,i)] = this.calculateMeanSquaredError(outData, trainData);
               
               this.hidden_log = outData;                   % log hidden unit activity for this pattern
               this.output_log = hiddenData;                % log output unit activity for this pattern
               [this.MSE_log(i),  this.MSE_patterns_log(:,i)] = this.calculateMeanSquaredError(outData, trainData);
               [this.CE_log(i), this.CE_patterns_log(:,i)] = this.calculateCrossEntropyError(outData, trainData);
               [this.CF_log(i), this.CF_patterns_log(:,i)] = this.calculateClassificationError(outData, trainData);
               [this.DimCE_log(i), this.DimCE_patterns_log(:,i)] = this.calculateDimClassificationError(outData, trainData);
               
               this.MSE_patterns_log(:,i) = MSE;            % calculate mean-squared error for whole set of patterns
               
               % stop learning if the mean-squared error reaches a certain
               % threshold
               if(this.MSE_log(i)) < this.thresh
                  break; 
               end
               
            end
            
        end

        % run through a data set (no training)
        function [outData, hiddenData, MSE, hidden_net, output_net, ceError, classError, classDimError] = runSet(this, varargin)
            
            % parse arguments: input patterns, task pattenrs, output patterns
            if(length(varargin) >=2)
                inputData = varargin{1};
                taskData = varargin{2};
            else
                inputData = this.inputSet;
                taskData = this.taskSet;
            end
            
            % check if input and task datasets have equal number of patterns (rows)
            if(size(inputData,1) ~= size(taskData,1))
                error('Task data has to have same number of rows as input data.');
            end
            
            Ndatasets = size(inputData,1);                  % total number of datasets
            
           % for each hidden layer 
            for hiddenLayer = 1:length(this.Nhidden)
    
                % calculate net inputs for hidden layer     
                if(length(this.Nhidden) > 1)
                    hidden_net_task = this.g(this.weights.W_TH{hiddenLayer}) * transpose(taskData);                              % input from task layer (task cue)
                    hidden_net_bias  = this.weights.W_BH{hiddenLayer} * (this.hidden_bias * ones(1,Ndatasets));   % input from hidden bias units
                else
                    hidden_net_task = this.g(this.weights.W_TH) * transpose(taskData);                              % input from task layer (task cue)
                    hidden_net_bias  = this.weights.W_BH * (this.hidden_bias * ones(1,Ndatasets));   % input from hidden bias units
                end
                
                if(hiddenLayer == 1)
                    hidden_net_input = this.weights.W_IH * transpose(inputData);                                    % input from input layer (stimulus)
                else
                    hidden_net_input = this.weights.W_HH{hiddenLayer-1} * this.hidden_act{hiddenLayer-1}; 
                end
                hidden_net{hiddenLayer} = hidden_net_input + hidden_net_task + hidden_net_bias;                              % total net input to hidden layer

                % calculate activation for hidden units
                if(length(this.Nhidden) > 1)
                    this.hidden_act{hiddenLayer} = 1./(1+exp(-hidden_net{hiddenLayer}));                                                      % use sigmoid activation function
                else
                    this.hidden_act = 1./(1+exp(-hidden_net));                                                      % use sigmoid activation function
                end

            end

           % calculate net input for output units
            output_net_task = this.g(this.weights.W_TO) * transpose(taskData);                              % input from task layer (task cue)
            if(length(this.Nhidden) > 1)
                output_net_hidden = this.weights.W_HO * this.hidden_act{end};                                        % input from hidden layer
            else
                output_net_hidden = this.weights.W_HO * this.hidden_act;                                        % input from hidden layer
            end
            output_net_bias   = this.weights.W_BO * (this.output_bias * ones(1,size(hidden_net_input,2)));  % input from output bias units
            output_net = output_net_hidden + output_net_task + output_net_bias;                             % total net input to output layer

            % calculate activation of output units
            this.output_act = 1./(1+exp(-output_net)); 

            % final network output
            if(length(this.Nhidden) > 1)
                for hiddenLayer = 1:length(this.hidden_act)
                    hiddenData{hiddenLayer} = this.hidden_act{hiddenLayer}';  % log activity of hidden units for each dataset
                end
            else
                    hiddenData = this.hidden_act';  % log activity of hidden units for each dataset
            end
            
            % hiddenData = this.hidden_act{end}';                      % log activity of hidden units for each dataset
            outData = this.output_act';                         % log activity of output units for each dataset
            if(length(this.Nhidden) > 1)
                for hiddenLayer = 1:length(this.Nhidden)
                    hidden_net{hiddenLayer} = hidden_net{hiddenLayer}';
                end
            else
                hidden_net= hidden_net';
            end
            output_net = output_net';
            
            % calculate MSE if correct output provided (train)
            MSE = -1*ones(1,Ndatasets);
            if(length(varargin)>=3)
                trainData = varargin{3};
                if(size(trainData,2) == size(outData,2))
                    % MSE
                    [~, MSE] = this.calculateMeanSquaredError(outData, trainData);
                    
                    % classification error
                    [~, classError] = this.calculateClassificationError(outData, trainData);
                    
                    % dimension classification error
                    [~, classDimError] = this.calculateDimClassificationError(outData, trainData);
                    
                    % cross-entropy error
                    [~, ceError] = this.calculateCrossEntropyError(outData, trainData);
                else
                    warning('Training data has to have same number of rows as input data. Can''t calculate MSE for each dataset.');
                end
            end
            
            % log activities for hidden and output units (for all patterns)
            this.hidden_log = hiddenData;
            this.output_log = outData;
            
        end
        
        
        % run a trial (feedforward step, no training)
        function [output_act, hidden_act, MSE, hidden_net, output_net] = runTrial(this, input, task, varargin)
            
            % initialize output activity
            output_act = zeros(1,this.Noutput);
            
            % for each hidden layer 
            for hiddenLayer = 1:length(this.Nhidden)
                
                if(length(this.Nhidden) > 1)
                    hidden_net_task = this.g(this.weights.W_TH{hiddenLayer}) * transpose(task);      % input from task layer (task cue)
                    hidden_net_bias  = this.weights.W_BH{hiddenLayer} * this.hidden_bias;            % input from hidden bias units
                else
                    hidden_net_task = this.g(this.weights.W_TH) * transpose(task);      % input from task layer (task cue)
                    hidden_net_bias  = this.weights.W_BH * this.hidden_bias;            % input from hidden bias units
                end
                
                % calculate net inputs for hidden layers
                if(hiddenLayer == 1)
                    hidden_net_input = this.weights.W_IH * transpose(input);            % input from input layer (stimulus)       
                else
                    hidden_net_input = this.weights.W_HH{hiddenLayer-1} * this.hidden_act{hiddenLayer-1}; % input from previous hidden layer  
                end
                hidden_net = hidden_net_input + hidden_net_task + hidden_net_bias;  % total net input to hidden layer
                
                % calculate activation for hidden units
                if(length(this.Nhidden) > 1)
                    this.hidden_act{hiddenLayer} = 1./(1+exp(-hidden_net));                          % use sigmoid activation function
                else
                    this.hidden_act = 1./(1+exp(-hidden_net));                          % use sigmoid activation function
                end
            end


            % calculate net input for output units
            output_net_task = this.g(this.weights.W_TO) * transpose(task);      % input from task layer (task cue)
            if(length(this.Nhidden) > 1)
                output_net_hidden = this.weights.W_HO * this.hidden_act{end};            % input from hidden layer
            else
                output_net_hidden = this.weights.W_HO * this.hidden_act;            % input from hidden layer
            end
            output_net_bias   = this.weights.W_BO * this.output_bias;           % input from output bias units
            output_net = output_net_hidden + output_net_task + output_net_bias; % total net input to output layer

            % calculate activation of output units
            this.output_act = 1./(1+exp(-output_net)); 

            % final network output
            output_act(:) = this.output_act';
            hidden_act = this.hidden_act;
            
            % calculate MSE if correct output provided (train)
            MSE = -1;
            if(~isempty(varargin))
                train = varargin{1};
                if(length(train) == length(output_act))
                    [~,  MSE] = this.calculateMeanSquaredError(this.output_act', train);
                end
            end
            
        end
        
        function [MSE_total, MSE_patterns] = calculateMeanSquaredError(this, varargin)
            
            outData = this.output_log;
            trainData = this.trainSet;
            
            if(~isempty(varargin))
                 outData = varargin{1};
            end
            
            if(length(varargin) > 1)
                 trainData = varargin{2};
            end
            
            if(~isequal(size(trainData), size(outData)))
                warning('Output data and training data has to have same dimensions in order to calculate classification error.');
            end
            
            MSE_patterns = sum((outData - trainData).^2, 2)./this.Noutput';
            MSE_total = mean(MSE_patterns);
            
        end
        
        function [CE_error_total, CE_error_patterns] = calculateCrossEntropyError(this, varargin)
            
            outData = this.output_log;
            trainData = this.trainSet;
            
            if(~isempty(varargin))
                 outData = varargin{1};
            end
            
            if(length(varargin) > 1)
                 trainData = varargin{2};
            end
            
            if(~isequal(size(trainData), size(outData)))
                warning('Output data and training data has to have same dimensions in order to calculate classification error.');
            end
            
            outData_normalized = outData./repmat(sum(outData,2),1,size(outData,2));
            CE_error_patterns = -sum(log(outData_normalized).*trainData,2);
            
            CE_error_total = mean(CE_error_patterns);
        end
        
        function [CF_error_total, CF_error_patterns] = calculateDimClassificationError(this, varargin)
            
            outData = this.output_log;
            trainData = this.trainSet;
            
            if(~isempty(varargin))
                 outData = varargin{1};
            end
            
            if(length(varargin) > 1)
                 trainData = varargin{2};
            end
            
            if(~isequal(size(trainData), size(outData)))
                warning('Output data and training data has to have same dimensions in order to calculate classification error.');
            end
            
            CF_error_patterns = nan(size(trainData,1),1);
            
            if(isempty(this.NPathways))
                warning('NPathways needs to be specified in order to calculate classificaiton error.');
            else
                
                % calculate number of features per output dimension
                NFeatures  = size(trainData,2)/this.NPathways;
                
                % Get the per pathway, max feature index, for each row of
                % data.
                curr_feat = 1:NFeatures;
                maxOut = zeros(size(outData,1), this.NPathways);
                maxTrain = zeros(size(outData,1), this.NPathways);
                for ii=1:this.NPathways
                    % For each row, in this pathway, get the max feature
                    % dim.
                    [maxV, maxOut(:,ii)] = max(outData(:, curr_feat), [], 2);
                    
                    % Now, do the same thing for the training signal.
                    % However, if the max feature dim is 0, then don't
                    % record that index, we will ignore it in the error
                    % calc.
                    [maxTrainV, maxTrain(:,ii)] = max(trainData(:, curr_feat), [], 2);
                    maxTrain(maxTrainV == 0,ii) = 0;
                     
                    curr_feat = curr_feat + NFeatures;
                end

                % Calculate the number of pathways that are not equal.
                % Ignore pathways for which the train pattern is all 0
                K = maxTrain ~= maxOut;
                K(maxTrain == 0) = 0;
                CF_error_patterns = sum(K, 2);
               
            end
            
            CF_error_total = mean(CF_error_patterns);
        end
        
        function [CF_error_total, CF_error_patterns] = calculateClassificationError(this, varargin)
            
            outData = this.output_log;
            trainData = this.trainSet;
            
            if(~isempty(varargin))
                 outData = varargin{1};
            end
            
            if(length(varargin) > 1)
                 trainData = varargin{2};
            end
            
            if(~isequal(size(trainData), size(outData)))
                warning('Output data and training data has to have same dimensions in order to calculate classification error.');
            end
        
            maxOutputs = max(outData,[],2);
            classes = zeros(size(outData));
            for i = 1:length(maxOutputs)
               classes(i,  outData(i,:) == maxOutputs(i)) = 1;
            end
            classError = sum(abs(classes-trainData),2);
            CF_error_patterns = zeros(length(classError),1);
            CF_error_patterns(classError > 0) = 1;
            
            CF_error_total = mean(CF_error_patterns);
        end
        
        % computes MSE only for tasks that are about to be executed
        % (ignores output dimensions that are not relevant)
        function [MSE_tasks_mean, MSE_tasks] = calculateMeanSquaredErrorTasks(this, varargin)
            
            outData = this.output_log;
            trainData = this.trainSet;
            taskData = this.taskSet;
            
            if(~isempty(varargin))
                 outData = varargin{1};
            end
            
            if(length(varargin) > 1)
                 trainData = varargin{2};
            end
            
            if(length(varargin) > 2)
                 taskData = varargin{3};
            end
            
            if(~isequal(size(trainData), size(outData)))
                warning('Output data and training data has to have same dimensions in order to calculate classification error.');
            end
            
            if(size(trainData,1) ~= size(taskData,1))
                warning('Output data and task data has to have same number of patterns (rows) order to calculate classification error.');
            end
            
            % compute task mask for MSE computation
            
            NFeatures = size(outData,2) / this.NPathways;
            
            MSE_tasks= nan(size(taskData));
            
            % compute mean MSE for invididual task being performed at a
            % given time
            for patternIdx = 1:size(taskData,1)
                
                tasks = find(taskData(patternIdx, :));
                for taskIdx = 1:length(tasks)
                    
                    task = tasks(taskIdx);
                    outputDim = mod(task-1,this.NPathways)+1;
                    relevantFeatures = (NFeatures*(outputDim-1)+1) : (NFeatures*outputDim);
                    MSE_tasks(patternIdx,task) = mean((outData(patternIdx, relevantFeatures) - trainData(patternIdx, relevantFeatures)).^2);

                end
                
            end
            
            % compute mean MSE for performed tasks
            MSE_tasks_mean = nanmean(MSE_tasks,2);
            
        end
      
        % computes absolute error only for tasks that are about to be executed
        % (ignores output dimensions that are not relevant)
        function [AE_tasks_mean, AE_tasks] = calculateAbsoluteErrorTasks(this, varargin)
            
            outData = this.output_log;
            trainData = this.trainSet;
            taskData = this.taskSet;
            
            if(~isempty(varargin))
                 outData = varargin{1};
            end
            
            if(length(varargin) > 1)
                 trainData = varargin{2};
            end
            
            if(length(varargin) > 2)
                 taskData = varargin{3};
            end
            
            if(~isequal(size(trainData), size(outData)))
                warning('Output data and training data has to have same dimensions in order to calculate classification error.');
            end
            
            if(size(trainData,1) ~= size(taskData,1))
                warning('Output data and task data has to have same number of patterns (rows) order to calculate classification error.');
            end
            
            % compute task mask for MSE computation
            
            NFeatures = size(outData,2) / this.NPathways;
            
            AE_tasks= nan(size(taskData));
            
            % compute mean MSE for invididual task being performed at a
            % given time
            for patternIdx = 1:size(taskData,1)
                
                tasks = find(taskData(patternIdx, :));
                for taskIdx = 1:length(tasks)
                    
                    task = tasks(taskIdx);
                    outputDim = mod(task-1,this.NPathways)+1;
                    relevantFeatures = (NFeatures*(outputDim-1)+1) : (NFeatures*outputDim);
                    AE_tasks(patternIdx,task) = 1 - mean(abs(outData(patternIdx, relevantFeatures) - trainData(patternIdx, relevantFeatures)));

                end
                
            end
            
            % compute mean MSE for performed tasks
            AE_tasks_mean = nanmean(AE_tasks,2);
            
        end
      
       % computes likelihood for each response for tasks that are about to
       % be executed by appying the softmax function to the output patterns
        % (ignores output dimensions that are not relevant). Also computes
        % general likelihood of correct response
        function [outcomeProbs, PCorrect_tasks_mean, PCorrect_tasks] = calculateSoftmaxProbabilitiesTasks(this, varargin)
            
            outData = this.output_log;
            trainData = this.trainSet;
            taskData = this.taskSet;
            
            if(~isempty(varargin))
                 outData = varargin{1};
            end
            
            if(length(varargin) > 1)
                 trainData = varargin{2};
            end
            
            if(length(varargin) > 2)
                 taskData = varargin{3};
            end
            
            if(~isequal(size(trainData), size(outData)))
                warning('Output data and training data has to have same dimensions in order to calculate classification error.');
            end
            
            if(size(trainData,1) ~= size(taskData,1))
                warning('Output data and task data has to have same number of patterns (rows) order to calculate classification error.');
            end
            
            % compute task mask for MSE computation
            
            NFeatures = size(outData,2) / this.NPathways;
            
            outcomeProbs =nan(size(outData));
            PCorrect_tasks= nan(size(taskData));
            PCorrect_tasks_mean= nan(size(PCorrect_tasks,1),1);
            
            % compute mean MSE for invididual task being performed at a
            % given time
            for patternIdx = 1:size(taskData,1)
                
                tasks = find(taskData(patternIdx, :));
                for taskIdx = 1:length(tasks)
                    
                    task = tasks(taskIdx);
                    
                    outputDim = mod(task-1,this.NPathways)+1;
                    relevantFeatures = (NFeatures*(outputDim-1)+1) : (NFeatures*outputDim);
                    correctResponse = relevantFeatures(find(trainData(patternIdx, relevantFeatures) == 1,1));
                    
                    % apply softmax
                    outcomeProbs(patternIdx,relevantFeatures) = exp(outData(patternIdx, relevantFeatures))./sum(exp(outData(patternIdx, relevantFeatures)));
                    
                    % compute correct response probability
                    PCorrect_tasks(patternIdx, task) = outcomeProbs(patternIdx, correctResponse);

                end
                
            end
            
            % compute mean MSE for performed tasks
            PCorrect_tasks_mean = nanmean(PCorrect_tasks,2);
            
        end
      
        % computes likelihood for each response for tasks that are about to
        % be executed by appying the softmax function to the output patterns
        % (ignores output dimensions that are not relevant). Also computes
        % general likelihood of correct response
        function [outcomeProbs, PCorrect_tasks_mean, PCorrect_tasks] = calculateOutcomeProbabilitiesTasks(this, varargin)
            
            outData = this.output_log;
            trainData = this.trainSet;
            taskData = this.taskSet;
            
            if(~isempty(varargin))
                 outData = varargin{1};
            end
            
            if(length(varargin) > 1)
                 trainData = varargin{2};
            end
            
            if(length(varargin) > 2)
                 taskData = varargin{3};
            end
            
            if(~isequal(size(trainData), size(outData)))
                warning('Output data and training data has to have same dimensions in order to calculate classification error.');
            end
            
            if(size(trainData,1) ~= size(taskData,1))
                warning('Output data and task data has to have same number of patterns (rows) order to calculate classification error.');
            end
            
            % compute task mask for MSE computation
            
            NFeatures = size(outData,2) / this.NPathways;
            
            outcomeProbs =nan(size(outData));
            PCorrect_tasks= nan(size(taskData));
            PCorrect_tasks_mean= nan(size(PCorrect_tasks,1),1);
            
            % compute mean MSE for invididual task being performed at a
            % given time
            for patternIdx = 1:size(taskData,1)
                
                tasks = find(taskData(patternIdx, :));
                
                for taskIdx = 1:length(tasks)
                    
                    task = tasks(taskIdx);
                    
                    outputDim = mod(task-1,this.NPathways)+1;
                    relevantFeatures = (NFeatures*(outputDim-1)+1) : (NFeatures*outputDim);
                    correctResponse = relevantFeatures(find(trainData(patternIdx, relevantFeatures) == 1,1));
                    
                    % apply softmax
                    outcomeProbs(patternIdx,relevantFeatures) = outData(patternIdx, relevantFeatures)./sum(outData(patternIdx, relevantFeatures));
                    
                    % compute correct response probability
                    PCorrect_tasks(patternIdx, task) = outcomeProbs(patternIdx, correctResponse);
                end
                
            end
            
            % compute mean MSE for performed tasks
            PCorrect_tasks_mean = nanmean(PCorrect_tasks,2);
        end
        
        
        
        
        
        
        
          % runs a leaky competitive accumulator for the response dimensions of each activated task
        function [optTaskAccuracy, optTaskRT, optAccuracy, optRT, optThreshIdx, RR, taskAccuracy, taskRT, taskRT_all, meanAccuracy, meanRT, outputProbabilities, RT_Simulations, noThresholdReachedPatternTaskPairs, minThresholdReachedPatternTaskPairs, maxThresholdReachedPatternTaskPairs] = runLCA(this, LCA_settings, varargin)
            
        % set input and output patterns
        if(length(varargin) >= 1) 
            input = varargin{1};
        else
            input = this.inputSet;
        end

        if(length(varargin) >= 2) 
            tasks = varargin{2};
        else
            tasks = this.taskSet;
        end

        if(length(varargin) >= 3) 
            train = varargin{3};
        else
            train = this.trainSet;
        end

         % check if NPathways is assigned
        if(isnan(this.NPathways)) 
            error('NNmodel.NPathways is not assigned. You need to specify the number of pathways of the NNmodel instance.');
        end

        % get number of features
        NFeatures = size(train,2)/this.NPathways;

        % default settings
        lambda_default = 0.2;
        alpha_default = 0.1;
        beta_default = 0.2;
        dt_tau_default = 0.1;
        c_default = 0.01;       % c_default = 1.58;
        maxTimeSteps_default = 1000;
        numSimulations_default = 1;
        W_ext_default = eye(NFeatures);
        T0_default = 0.15;
        effectiveLeakage_default = 0.2;
        responseThreshold_default = 0.2;
        tau_default = 0.1;
        dt_default = 0.01;
        optimizeTasksWithinPattern_default = 1;
        showWarning = 0;

        % set LCA parameters

        if(isfield(LCA_settings, 'lambda'))
            lambda = LCA_settings.lambda;
        else
            lambda = lambda_default;
            if(showWarning) 
                warning(['No lambda specified, will use default value of ' num2str(lambda)]);
            end
        end

        if(isfield(LCA_settings, 'alpha'))
            alpha = LCA_settings.alpha;
        else
            alpha = alpha_default;
            if(showWarning) 
                warning(['No alpha specified, will use default value of ' num2str(alpha)]);
            end
        end

        if(isfield(LCA_settings, 'beta'))
            beta = LCA_settings.beta;
        else
            beta = beta_default;
            if(showWarning) 
                warning(['No beta specified, will use default value of ' num2str(beta)]);
            end
        end

        if(isfield(LCA_settings, 'tau'))
            tau = LCA_settings.tau;
        else
            tau = tau_default;
            if(showWarning) 
                warning(['No tau specified, will use default value of ' num2str(tau)]);
            end
        end

        if(isfield(LCA_settings, 'dt'))
            dt = LCA_settings.dt;
        else
            dt = dt_default;
            if(showWarning) 
                warning(['No dt specified, will use default value of ' num2str(dt)]);
            end
        end

        if(isfield(LCA_settings, 'dt_tau'))
            dt_tau = LCA_settings.dt_tau;
            dt = dt_tau * tau;
        else
            dt_tau = dt_tau_default;
            if(showWarning) 
                warning(['No dt_tau specified, will use default value of ' num2str(dt_tau)]);
            end
        end

        if(isfield(LCA_settings, 'c'))
            c = LCA_settings.c;
        else
            c = c_default;
            if(showWarning) 
                warning(['No c specified, will use default value of ' num2str(c)]);
            end
        end

         if(isfield(LCA_settings, 'responseThreshold'))
            responseThreshold = LCA_settings.responseThreshold;
        else
            responseThreshold = responseThreshold_default;
            if(showWarning) 
                warning(['No responseThreshold specified, will use default value of ' num2str(responseThreshold)]);
            end
        end

        if(isfield(LCA_settings, 'maxTimeSteps'))
            maxTimeSteps = LCA_settings.maxTimeSteps;
        else
            maxTimeSteps = maxTimeSteps_default;
            if(showWarning) 
                warning(['No maxTimeSteps specified, will use default value of ' num2str(maxTimeSteps)]);
            end
        end

        if(isfield(LCA_settings, 'numSimulations'))
            numSimulations = LCA_settings.numSimulations;
        else
            numSimulations = numSimulations_default;
            if(showWarning) 
                warning(['No numSimulations specified, will use default value of ' num2str(numSimulations)]);
            end
        end

        if(isfield(LCA_settings, 'W_ext'))
            W_ext = LCA_settings.W_ext;
        else
            W_ext = W_ext_default;
            if(showWarning) 
                warning(['No W_ext specified, will use default value']);
            end
        end

        if(isfield(LCA_settings, 'T0'))
            T0 = LCA_settings.T0;
        else
            T0 = T0_default;
            if(showWarning) 
                warning(['No T0 specified, will use default value of ' num2str(T0)]);
            end
        end

        if(isfield(LCA_settings, 'effectiveLeakage'))
            effectiveLeakage = LCA_settings.effectiveLeakage;
        else
            if(isfield(LCA_settings, 'lambda') && isfield(LCA_settings, 'alpha'))
                effectiveLeakage = lambda - alpha;
            else
                effectiveLeakage = effectiveLeakage_default;
                if(showWarning) 
                    warning(['No effectiveLeakage specified, will use default value of ' num2str(effectiveLeakage)]);
                end
            end
        end

        if(isfield(LCA_settings, 'optimizeTasksWithinPattern'))
            optimizeTasksWithinPattern = LCA_settings.optimizeTasksWithinPattern;
        else
            optimizeTasksWithinPattern = optimizeTasksWithinPattern_default;
        end

        % compute output patterns
        [output] = this.runSet(input, tasks, train);

        % generate output matrix
        numRespPatterns = sum(tasks(:) == 1);
        numPatterns = size(output,1);

        LCA_input = nan(numRespPatterns, NFeatures); % matrix that contains each response dimension
        LCA_train = nan(numRespPatterns, NFeatures);
        LCA_patternIdx =  nan(numRespPatterns, 1);
        LCA_tasksIdx =  nan(numRespPatterns, 1);

        % fill in new output matrix
        inputCounter = 1;
        for patternIdx = 1:size(output,1)

            currentTasks = find(tasks(patternIdx,:) == 1);

            for taskIdx = 1:length(currentTasks)

                currentTask = currentTasks(taskIdx);
                currentOutputDim = mod(currentTask-1, this.NPathways) + 1;
                outputFeatures = ((currentOutputDim-1) * NFeatures + 1) : (currentOutputDim * NFeatures);

                LCA_input(inputCounter, :) = output(patternIdx, outputFeatures);
                LCA_train(inputCounter, :) = train(patternIdx, outputFeatures);
                LCA_patternIdx(inputCounter, :) = patternIdx;
                LCA_tasksIdx(inputCounter, :) = currentTask;     % taskIdx

                inputCounter = inputCounter +1;

            end

        end
        
        % initialize each LCA
        x = zeros([size(LCA_input) numSimulations maxTimeSteps]); % net input
        f = zeros([size(LCA_input) numSimulations maxTimeSteps]); % activation

        % generate noise for all LCAs and timesteps
        noise = c.*randn([size(LCA_input) numSimulations maxTimeSteps]);

        % compute external input I_ext = p
        p = LCA_input * W_ext;
        p = repmat(p, [1, 1, numSimulations]);

        % compute lateral inhibition matrix
        W_inhib = -1 * ones(NFeatures, NFeatures) + eye(NFeatures, NFeatures);

        % compute sqrt(tau)
        sqrt_dt_tau = sqrt(dt_tau);

        % iterate through each time step
        for t = 1:maxTimeSteps

            % initialize leaky accumulator units
            if(t==1)
                x(:,:,:,t) = 0;
            else
                % compute new accumulator net input

                x_t = squeeze(x(:,:,:,t-1));    % current accumulator net input
                f_t = squeeze(f(:,:,:,t-1));

                % compute lateral inhibition between accumulator units
                lateralInhibition = nan(size(x_t));

                for i = 1:size(x_t, 3)
                    lateralInhibition(:,:,i) = beta * squeeze(f_t(:,:,i)) * W_inhib;  
                end

                dx = (p - lambda * x_t + alpha * f_t + lateralInhibition) * dt_tau + squeeze(noise(:,:,:,t)) * sqrt_dt_tau; % see Eqn. 3 in Usher & McClelland (2001)
                x(:,:,:,t) = x_t + dx;              % new accumulator net input

            end

            % compute activation
            f_t = x(:,:,:,t);
            f_t(f_t < 0) = 0;
            f(:,:,:,t) = f_t;

        end

        %             inputToPlot = 1;
        %             stepsToPlot = 50;
        %             figure(1);
        %             disp(LCA_input(inputToPlot,:));
        %             plot(squeeze(f(inputToPlot,1,1,1:stepsToPlot)), 'b'); hold on;
        %             plot(squeeze(f(inputToPlot,2,1,1:stepsToPlot)), 'r'); hold off;

        % determine number of thresholds
        numThresholds = length(responseThreshold);

        % initialize outcome probabilities and reaction times for each simulation
        RT_Simulations = nan(numThresholds, numRespPatterns, numSimulations);
        P_Outcome_Simulations = zeros(numThresholds, numRespPatterns, NFeatures, numSimulations);

%         responseThreshold
%         squeeze(f(3, :, 1, :))
        

        % determine maximum reaction time
        maxRT = maxTimeSteps * dt + T0;
        
        % log all pattern-task pairs for which response threshild is not reached
        noThresholdReachedPatternTaskPairs = [];
        
        % determine RT and outcome probability for each accumulator unit
        for patternIdx = 1:numRespPatterns

            for simIdx = 1:numSimulations

                for thresholdIdx = 1:numThresholds

                    threshold = responseThreshold(thresholdIdx);

                    RT_features = zeros(1, NFeatures);

                    % get time points of crossing thresholds
                    for featureIdx = 1:NFeatures
                        timeCourse = squeeze(f(patternIdx, featureIdx, simIdx, :));
                        threshIdx = find(timeCourse > threshold);

                        if(~isempty(threshIdx))
                            RT_features(featureIdx) = threshIdx(1);
                        else
                            RT_features(featureIdx) = nan;
                        end
                    end

                    % compute RT and outcome
                    if(all(isnan(RT_features)))
                        % no threshold hit
                        RT_Simulations(thresholdIdx, patternIdx, simIdx) = maxRT;
                    else
                        RT_Simulations(thresholdIdx, patternIdx, simIdx) = min(RT_features)*dt + T0;
                    end
                    
                    % we only allow one feature to pass threshold
                    winningFeature = find(RT_features == min(RT_features));
                    if(length(winningFeature) > 1)
                        
                        % if multiple features pass threhsold then randomly select one
                        winningFeature = winningFeature(randperm(length(winningFeature)));
                        winningFeature = winningFeature(1);

                    end
                    
                    P_Outcome_Simulations(thresholdIdx, patternIdx, winningFeature, simIdx) = 1;

                end
                
            end
            
           allPatternMaxRTs = max(RT_Simulations(:, patternIdx, :),[], 3);
           if(all (allPatternMaxRTs == maxRT))
               warning(['No response threshold reached for task ' num2str(LCA_tasksIdx(patternIdx)) ' in pattern ' num2str(LCA_patternIdx(patternIdx))]);
               noThresholdReachedPatternTaskPairs = [noThresholdReachedPatternTaskPairs; [LCA_patternIdx(patternIdx) LCA_tasksIdx(patternIdx)]];
           end
           
%            disp(['pattern ' num2str(patternIdx) '/' num2str(numRespPatterns)]);
           
        end

        % determine outcome probabilities and reaction times
        if(numThresholds > 1)
            RTs = mean(RT_Simulations, 3);
        else
            RTs(1, :) = mean(RT_Simulations, 3);
        end

        % determine accuracies
        P_Outcome = mean(P_Outcome_Simulations, 4);
        P_Outcome = permute(P_Outcome, [1 3 2]); % permute for indexing (from LCA_train)
        
        correctUnitsIndicies = find(transpose(LCA_train )== 1);

        if(ndims(P_Outcome) == 3)
            Accuracies = squeeze(P_Outcome(:, correctUnitsIndicies));
        else
            Accuracies = squeeze(P_Outcome(correctUnitsIndicies));
        end

        % initialize final outputs
        taskAccuracy = nan(numThresholds, numPatterns, size(tasks,2));
        taskRT = nan(numThresholds, numPatterns, size(tasks,2));
        taskRT_all = nan(numThresholds, numPatterns, size(tasks,2), numSimulations);
        outputProbabilities = nan(numThresholds, size(train,1), size(train,2));

        % format final outputs
        for threshIdx = 1:numThresholds

            outputCounter = 1;

            for patternIdx = 1:size(output,1)

                currentTasks = find(tasks(patternIdx,:) == 1);

                for taskIdx = 1:length(currentTasks)

                    currentTask = currentTasks(taskIdx);
                    currentOutputDim = mod(currentTask-1, this.NPathways) + 1;
                    outputFeatures = ((currentOutputDim-1) * NFeatures + 1) : (currentOutputDim * NFeatures);

                    taskAccuracy(threshIdx, patternIdx, currentTask) = Accuracies(threshIdx, outputCounter);
                    taskRT(threshIdx, patternIdx, currentTask) = RTs(threshIdx, outputCounter);
                    taskRT_all(threshIdx, patternIdx, currentTask, :) = RT_Simulations(threshIdx, outputCounter, :);
                    outputProbabilities(threshIdx, patternIdx, outputFeatures) = P_Outcome(threshIdx, :, outputCounter);

                    outputCounter = outputCounter + 1;

                end

                
            end

        end
        
        meanAccuracy = nanmean(taskAccuracy, 3);
        meanRT = nanmean(taskRT, 3);

         % determine optimal responses based on reward rate (RR)
         optTaskAccuracy = nan(size(taskAccuracy,2), size(taskAccuracy,3));
         optTaskRT= nan(size(taskRT,2), size(taskRT,3));
         optAccuracy = nan(1, size(meanAccuracy, 2));
         optRT = nan(1, size(meanRT, 2));
         
         % pairs of patterns and tasks for which a boundary threshold is found
         minThresholdReachedPatternTaskPairs = [];
         maxThresholdReachedPatternTaskPairs = [];

        if(optimizeTasksWithinPattern)
            % optimize RR for each single task within a pattern
            RR = taskAccuracy ./ taskRT;

            optThreshIdx = nan(size(taskAccuracy, 2), size(taskAccuracy, 3));
            relevantTasks = find(tasks == 1);
            [row, col] = ind2sub(size(tasks), relevantTasks);

            for task = 1:length(relevantTasks)

                patternIdx = row(task);
                taskIdx = col(task);

                optThresh = find(RR(:, patternIdx, taskIdx) == max(RR(:, patternIdx, taskIdx)));

                if(isempty(optThresh))
                    optThresh = 1;
                    warning(['Could not find optimal threshold for task ' num2str(taskIdx) ' in pattern ' num2str(patternIdx) '. Consider increasing number of maximum time steps.']);
                end
                
                if(responseThreshold(optThresh) == min(responseThreshold) & min(responseThreshold) > 0)
                    warning(['Picked minimum threshold for task ' num2str(taskIdx) ' in pattern ' num2str(patternIdx) '. Consider increasing range of tested thresholds.']);
                    minThresholdReachedPatternTaskPairs = [minThresholdReachedPatternTaskPairs; patternIdx taskIdx];
                end
                
                if(responseThreshold(optThresh) == max(responseThreshold))
                    warning(['Picked maximum threshold for task ' num2str(taskIdx) ' in pattern ' num2str(patternIdx) '. Consider increasing range of tested thresholds.']);
                    maxThresholdReachedPatternTaskPairs = [maxThresholdReachedPatternTaskPairs; patternIdx taskIdx];
                end

                optThreshIdx(patternIdx, taskIdx) = optThresh(1);
                optTaskAccuracy(patternIdx, taskIdx) = taskAccuracy(optThresh(1), patternIdx, taskIdx);
                optTaskRT(patternIdx, taskIdx) = taskRT(optThresh(1), patternIdx, taskIdx);
            end

            optAccuracy = nanmean(optTaskAccuracy, 2);
            optRT = nanmean(optTaskRT, 2);

        else
            % optimize RR for each pattern
            RR = meanAccuracy ./ meanRT;

            optThreshIdx = nan(1, size(meanAccuracy,2));
            for patternIdx = 1:size(meanAccuracy, 2)
                optThresh = find(RR(:, patternIdx) == max(RR(:, patternIdx)));

                 if(isempty(optThresh))
                    optThresh = 1;
                    warning(['Could not find optimal threshold for pattern ' num2str(patternIdx) '. Consider increasing number of maximum time steps.']);
                 elseif(optThresh == 1)
                         warning(['Reached lower bound of suggested thresholds.']);
                 elseif (optThresh == numThresholds)
                         warning(['Reached upper bound of suggested thresholds.']);
                 end
                     
                optThreshIdx(patternIdx) = optThresh(1);
                optTaskAccuracy(patternIdx, :) = taskAccuracy(optThresh(1), patternIdx, :);
                optTaskRT(patternIdx, :) = taskRT(optThresh(1), patternIdx, :);
                optAccuracy(patternIdx) = meanAccuracy(optThresh(1), patternIdx);
                optRT(patternIdx) = meanRT(optThresh(1), patternIdx);

            end

        end
            
        end
       
         % runs a leaky competitive accumulator for the response dimensions of each activated task
        function [optTaskAccuracy, optTaskRT, optAccuracy, optRT, optThreshIdx, RR, taskAccuracy, taskRT, meanAccuracy, meanRT, outputProbabilities] = runLCA_RDist(this, LCA_settings, varargin)
            
        %addpath('RDist');    
            
        % set input and output patterns
        if(length(varargin) >= 1) 
            input = varargin{1};
        else
            input = this.inputSet;
        end

        if(length(varargin) >= 2) 
            tasks = varargin{2};
        else
            tasks = this.taskSet;
        end

        if(length(varargin) >= 3) 
            train = varargin{3};
        else
            train = this.trainSet;
        end

         % check if NPathways is assigned
        if(isnan(this.NPathways)) 
            error('NNmodel.NPathways is not assigned. You need to specify the number of pathways of the NNmodel instance.');
        end

        % get number of features
        NFeatures = size(train,2)/this.NPathways;

        % default settings
        lambda_default = 0.2;
        alpha_default = 0.1;
        beta_default = 0.2;
        dt_tau_default = 0.1;
        c_default = 0.01;       % c_default = 1.58;
        maxTimeSteps_default = 1000;
        numSimulations_default = 1;
        W_ext_default = eye(NFeatures);
        T0_default = 0.15;
        effectiveLeakage_default = 0.2;
        responseThreshold_default = 0.2;
        tau_default = 0.1;
        dt_default = 0.01;
        optimizeTasksWithinPattern_default = 1;
        showWarning = 0;

        % set LCA parameters

        if(isfield(LCA_settings, 'lambda'))
            lambda = LCA_settings.lambda;
        else
            lambda = lambda_default;
            if(showWarning) 
                warning(['No lambda specified, will use default value of ' num2str(lambda)]);
            end
        end

        if(isfield(LCA_settings, 'alpha'))
            alpha = LCA_settings.alpha;
        else
            alpha = alpha_default;
            if(showWarning) 
                warning(['No alpha specified, will use default value of ' num2str(alpha)]);
            end
        end

        if(isfield(LCA_settings, 'beta'))
            beta = LCA_settings.beta;
        else
            beta = beta_default;
            if(showWarning) 
                warning(['No beta specified, will use default value of ' num2str(beta)]);
            end
        end

        if(isfield(LCA_settings, 'tau'))
            tau = LCA_settings.tau;
        else
            tau = tau_default;
            if(showWarning) 
                warning(['No tau specified, will use default value of ' num2str(tau)]);
            end
        end

        if(isfield(LCA_settings, 'dt'))
            dt = LCA_settings.dt;
        else
            dt = dt_default;
            if(showWarning) 
                warning(['No dt specified, will use default value of ' num2str(dt)]);
            end
        end

        if(isfield(LCA_settings, 'dt_tau'))
            dt_tau = LCA_settings.dt_tau;
            dt = dt_tau * tau;
        else
            dt_tau = dt_tau_default;
            if(showWarning) 
                warning(['No dt_tau specified, will use default value of ' num2str(dt_tau)]);
            end
        end

        if(isfield(LCA_settings, 'c'))
            c = LCA_settings.c;
        else
            c = c_default;
            if(showWarning) 
                warning(['No c specified, will use default value of ' num2str(c)]);
            end
        end

         if(isfield(LCA_settings, 'responseThreshold'))
            responseThreshold = LCA_settings.responseThreshold;
        else
            responseThreshold = responseThreshold_default;
            if(showWarning) 
                warning(['No responseThreshold specified, will use default value of ' num2str(responseThreshold)]);
            end
        end

        if(isfield(LCA_settings, 'maxTimeSteps'))
            maxTimeSteps = LCA_settings.maxTimeSteps;
        else
            maxTimeSteps = maxTimeSteps_default;
            if(showWarning) 
                warning(['No maxTimeSteps specified, will use default value of ' num2str(maxTimeSteps)]);
            end
        end

        if(isfield(LCA_settings, 'numSimulations'))
            numSimulations = LCA_settings.numSimulations;
        else
            numSimulations = numSimulations_default;
            if(showWarning) 
                warning(['No numSimulations specified, will use default value of ' num2str(numSimulations)]);
            end
        end

        if(isfield(LCA_settings, 'W_ext'))
            W_ext = LCA_settings.W_ext;
        else
            W_ext = W_ext_default;
            if(showWarning) 
                warning(['No W_ext specified, will use default value']);
            end
        end

        if(isfield(LCA_settings, 'T0'))
            T0 = LCA_settings.T0;
        else
            T0 = T0_default;
            if(showWarning) 
                warning(['No T0 specified, will use default value of ' num2str(T0)]);
            end
        end

        if(isfield(LCA_settings, 'effectiveLeakage'))
            effectiveLeakage = LCA_settings.effectiveLeakage;
        else
            if(isfield(LCA_settings, 'lambda') && isfield(LCA_settings, 'alpha'))
                effectiveLeakage = lambda - alpha;
            else
                effectiveLeakage = effectiveLeakage_default;
                if(showWarning) 
                    warning(['No effectiveLeakage specified, will use default value of ' num2str(effectiveLeakage)]);
                end
            end
        end

        if(isfield(LCA_settings, 'optimizeTasksWithinPattern'))
            optimizeTasksWithinPattern = LCA_settings.optimizeTasksWithinPattern;
        else
            optimizeTasksWithinPattern = optimizeTasksWithinPattern_default;
        end
         
        % compute output patterns
        [output] = this.runSet(input, tasks, train);

        % generate output matrix
        numRespPatterns = sum(tasks(:) == 1);
        numPatterns = size(output,1);

        LCA_input = nan(numRespPatterns, NFeatures); % matrix that contains each response dimension
        LCA_train = nan(numRespPatterns, NFeatures);
        LCA_patternIdx =  nan(numRespPatterns, 1);
        LCA_tasksIdx =  nan(numRespPatterns, 1);

        % fill in new output matrix
        inputCounter = 1;
        for patternIdx = 1:size(output,1)

            currentTasks = find(tasks(patternIdx,:) == 1);

            for taskIdx = 1:length(currentTasks)

                currentTask = currentTasks(taskIdx);
                currentOutputDim = mod(currentTask-1, this.NPathways) + 1;
                outputFeatures = ((currentOutputDim-1) * NFeatures + 1) : (currentOutputDim * NFeatures);

                LCA_input(inputCounter, :) = output(patternIdx, outputFeatures);
                LCA_train(inputCounter, :) = train(patternIdx, outputFeatures);
                LCA_patternIdx(inputCounter, :) = patternIdx;
                LCA_tasksIdx(inputCounter, :) = currentTask;     % taskIdx

                inputCounter = inputCounter +1;

            end

        end
        
        
        % set up LCA Dist Wrapper
        
        % define simulation variables
        if(~length(varargin) >= 4)
            settings= varargin{4};
        else
            settings.nWalkers = numSimulations; % num simulations
            settings.seed = 0;              % random seed is clock dependent
            settings.dt = dt;        % time step Euler Maruyama method
            settings.nBins = 3000;      % number of bins constituting the simulated RT probability distribution
            settings.binWidth = 0.001;      % RT di erence spanned by each bin   
            settings.nGPUs = 0;         % number of assigned GPUs (0 uses CPU)
        end
        
        [settings,ok]= checkSettings(settings, 1);

        if(~ok)
            error('LCA Wrapper settings variable not ok.');
        end
        
        % define LCA variables
        if(~length(varargin) >= 5)
            LCAParams= varargin{5};
        else
            LCAParams.nDim = size(LCA_input, 2);    % dimensionality evidence space
            LCAParams.nStimuli = size(LCA_input, 1); % number of stimuli
            LCAParams.c = ones(LCAParams.nDim, 1) *c;               % diffusion constant (noise)
            LCAParams.a = ones(LCAParams.nDim, 1) * 0.5;              % evidence thresholds
            LCAParams.startPos = ones(LCAParams.nDim, 1) * 0.0001;              % evidence thresholds
            LCAParams.v = transpose(LCA_input) .* W_ext(1);          % drift rates induced by the stimuli (constant coe cients of SDEs)
            LCAParams.Ter = T0;          % mean non-decision time
            LCAParams.sTer = 0;         % width of uniform non-decision time distribution
            LCAParams.profile = 1;      % time-varying multiplier of drift rate
            LCAParams.Gamma = ones(LCAParams.nDim) * beta;      % linear coe cients of SDEs (diagonal: self, off-diagonal: mutual)
            LCAParams.Gamma(eye(LCAParams.nDim) == 1) = effectiveLeakage; 
        end
        
        [LCAParams,ok]= checkLCAPars(LCAParams, 1);

        if(~ok)
            error('LCA parameters not ok.');
        end
        
        % determine number of thresholds
        numThresholds = length(responseThreshold);
        
        RTs = nan(numThresholds, LCAParams.nStimuli);
        P_Outcome = nan(numThresholds, LCAParams.nDim,  LCAParams.nStimuli);
        
        % compute RTs and outcome probabilities for all thresholds
        for thresholdIdx = 1:numThresholds
            
            threshold = responseThreshold(thresholdIdx);
            
            % Can't run RTDist with 'a' paramters too small (0 is to small)
            if threshold == 0
               continue 
            end
            
            LCAParams.a = ones(LCAParams.nDim, 1) * threshold;
            
            % run LCA
            rng('default')
            [distributions, ok] = LCADist(LCAParams,settings, 1);
            
            if(~ok);
                warning('LCADist call not ok.');
            end
            
            % reshape RT distrinbutions
            RTBins = single(reshape(distributions, [size(distributions,1), LCAParams.nDim, LCAParams.nStimuli]));
            
            % size(RTBins, 1) ... # RT bins
            % size(RTBins, 2) ... # features per output dimension
            % size(RTBins, 3) ... # patterns
            
            % generate cumulative RTs
            cumRTs = transpose(linspace(0, settings.dt, size(RTBins, 1)));
            cumRTs = repmat(cumRTs, [1. size(RTBins,2), size(RTBins, 3)]);
            
            RT_binned = RTBins .* cumRTs; % dim: RT bins x features per output dimension x patterns
            RTs(thresholdIdx,:) = squeeze(mean(mean(RT_binned)));
            
            % compute accuracies
            BinSums = squeeze(sum(RTBins,1));
            BinsPerPattern = repmat(sum(BinSums,1), size(RTBins,2), 1);
            
            P_Outcome(thresholdIdx,:,:) = BinSums ./ BinsPerPattern;

        end

        correctUnitsIndicies = find(transpose(LCA_train )== 1);

        if(ndims(P_Outcome) == 3)
            Accuracies = squeeze(P_Outcome(:, correctUnitsIndicies));
        else
            Accuracies = squeeze(P_Outcome(correctUnitsIndicies));
        end

        % initialize final outputs
        taskAccuracy = nan(numThresholds, numPatterns, size(tasks,2));
        taskRT = nan(numThresholds, numPatterns, size(tasks,2));
        taskRT_all = nan(numThresholds, numPatterns, size(tasks,2), numSimulations);
        outputProbabilities = nan(numThresholds, size(train,1), size(train,2));

        % format final outputs
        for threshIdx = 1:numThresholds

            outputCounter = 1;

            for patternIdx = 1:size(output,1)

                currentTasks = find(tasks(patternIdx,:) == 1);

                for taskIdx = 1:length(currentTasks)

                    currentTask = currentTasks(taskIdx);
                    currentOutputDim = mod(currentTask-1, this.NPathways) + 1;
                    outputFeatures = ((currentOutputDim-1) * NFeatures + 1) : (currentOutputDim * NFeatures);

                    taskAccuracy(threshIdx, patternIdx, currentTask) = Accuracies(threshIdx, outputCounter);
                    taskRT(threshIdx, patternIdx, currentTask) = RTs(threshIdx, outputCounter);
                    outputProbabilities(threshIdx, patternIdx, outputFeatures) = P_Outcome(threshIdx, :, outputCounter);

                    outputCounter = outputCounter + 1;

                end

                
            end

        end
        

        meanAccuracy = nanmean(taskAccuracy, 3);
        meanRT = nanmean(taskRT, 3);

         % determine optimal responses based on reward rate (RR)
         optTaskAccuracy = nan(size(taskAccuracy,2), size(taskAccuracy,3));
         optTaskRT= nan(size(taskRT,2), size(taskRT,3));
         optAccuracy = nan(1, size(meanAccuracy, 2));
         optRT = nan(1, size(meanRT, 2));
         
         % pairs of patterns and tasks for which a boundary threshold is found
         minThresholdReachedPatternTaskPairs = [];
         maxThresholdReachedPatternTaskPairs = [];

        if(optimizeTasksWithinPattern)
            % optimize RR for each single task within a pattern
            RR = taskAccuracy ./ taskRT;

            optThreshIdx = nan(size(taskAccuracy, 2), size(taskAccuracy, 3));
            relevantTasks = find(tasks == 1);
            [row, col] = ind2sub(size(tasks), relevantTasks);

            for task = 1:length(relevantTasks)

                patternIdx = row(task);
                taskIdx = col(task);

                optThresh = find(RR(:, patternIdx, taskIdx) == max(RR(:, patternIdx, taskIdx)));

                if(isempty(optThresh))
                    optThresh = 1;
                    warning(['Could not find optimal threshold for task ' num2str(taskIdx) ' in pattern ' num2str(patternIdx) '. Consider increasing number of maximum time steps.']);
                end
                
                if(responseThreshold(optThresh) == min(responseThreshold))
                    warning(['Picked minimum threshold for task ' num2str(taskIdx) ' in pattern ' num2str(patternIdx) '. Consider increasing range of tested thresholds.']);
                    minThresholdReachedPatternTaskPairs = [minThresholdReachedPatternTaskPairs; patternIdx taskIdx];
                end
                
                if(responseThreshold(optThresh) == max(responseThreshold))
                    warning(['Picked maximum threshold for task ' num2str(taskIdx) ' in pattern ' num2str(patternIdx) '. Consider increasing range of tested thresholds.']);
                    maxThresholdReachedPatternTaskPairs = [maxThresholdReachedPatternTaskPairs; patternIdx taskIdx];
                end

                optThreshIdx(patternIdx, taskIdx) = optThresh(1);
                optTaskAccuracy(patternIdx, taskIdx) = taskAccuracy(optThresh(1), patternIdx, taskIdx);
                optTaskRT(patternIdx, taskIdx) = taskRT(optThresh(1), patternIdx, taskIdx);
            end

            optAccuracy = nanmean(optTaskAccuracy, 2);
            optRT = nanmean(optTaskRT, 2);

        else
            % optimize RR for each pattern
            RR = meanAccuracy ./ meanRT;

            optThreshIdx = nan(1, size(meanAccuracy,2));
            for patternIdx = 1:size(meanAccuracy, 2)
                optThresh = find(RR(:, patternIdx) == max(RR(:, patternIdx)));

                 if(isempty(optThresh))
                    optThresh = 1;
                    warning(['Could not find optimal threshold for pattern ' num2str(patternIdx) '. Consider increasing number of maximum time steps.']);
                 elseif(optThresh == 1)
                         warning(['Reached lower bound of suggested thresholds.']);
                 elseif (optThresh == numThresholds)
                         warning(['Reached upper bound of suggested thresholds.']);
                 end
                     
                optThreshIdx(patternIdx) = optThresh(1);
                optTaskAccuracy(patternIdx, :) = taskAccuracy(optThresh(1), patternIdx, :);
                optTaskRT(patternIdx, :) = taskRT(optThresh(1), patternIdx, :);
                optAccuracy(patternIdx) = meanAccuracy(optThresh(1), patternIdx);
                optRT(patternIdx) = meanRT(optThresh(1), patternIdx);

            end

        end
            
        end
        
        
        
        
        
        
        
        
        
        
      
        % calculate switch dynamics between inputs based on delayed net
        % input
        function [output_act_log, hidden_act_log, MSE, DDM_RTs, DDM_ERs, E, A] = switchTasks(this, tau, iterations, inputA, inputB, tasksA, tasksB, varargin)
            
            if(length(varargin) >=1)
                trainA = varargin{1};
            else
                trainA = [];
            end
            
            if(length(varargin) >=2)
                trainB = varargin{2};
            else
                trainB = [];
            end
            
            if(length(varargin) >=3)
                ddmp = varargin{3};
                runAccumulator = 1;
            else
                ddmp = [];
                runAccumulator = 0;
            end
            
            if(iterations < 2)
                warning('Number of iterations should be at least 2 in order to simulate a task switch.');
            end
            
            if(~isequal(size(inputA), size(inputB)))
               error('Dimensions of input patterns need to match.'); 
            end
            
            if(~isequal(size(tasksA), size(tasksB)))
               error('Dimensions of task patterns need to match.'); 
            end
            
            if(~isequal(size(inputA,1), size(tasksA,1)))
               error('Number of input and task patterns need to match.'); 
            end
            
            if(~isequal(size(inputA,1), size(trainB,1)))
               error('Number of input patterns and correct output patterns need to match.'); 
            end
            
            % prepare log data
            Nsets = size(inputA,1);
            output_net_log = nan(Nsets, iterations, this.Noutput);
            output_act_log = nan(Nsets, iterations, this.Noutput);
            hidden_net_log = nan(Nsets, iterations, this.Nhidden);
            hidden_act_log = nan(Nsets, iterations, this.Nhidden);
            MSE = nan(Nsets, iterations);
            
            % get first data
            [output_act_A, hidden_act_A, MSE_A, hidden_net_A, output_net_A] = runSet(this, inputA, tasksA, trainA);

            output_net_log(:,1,:) = output_net_A;
            output_act_log(:,1,:) = output_act_A;
            hidden_net_log(:,1,:) = hidden_net_A;
            hidden_act_log(:,1,:) = hidden_act_A;

            MSE(:,1) = MSE_A;
                
            for set = 1:Nsets
                
                input = inputB(set,:);
                task = tasksB(set,:);
                train = trainB(set,:);
                
                % loop through time steps
                for i = 2:iterations

                    % calculate net inputs for hidden layer
                    hidden_net_input = this.weights.W_IH * transpose(input);            % input from input layer (stimulus)       
                    hidden_net_task = this.g(this.weights.W_TH) * transpose(task);      % input from task layer (task cue)
                    hidden_net_bias  = this.weights.W_BH * this.hidden_bias;            % input from hidden bias units
                    hidden_net = hidden_net_input + hidden_net_task + hidden_net_bias;  % total net input to hidden layer
                    hidden_net_log(set,i,:) = tau*hidden_net + (1-tau) * squeeze(hidden_net_log(set,i-1,:)); % integrate input from previous time step and current time step
                    hidden_net = squeeze(hidden_net_log(set,i,:));
                    
                    % calculate activation for hidden units
                    this.hidden_act = 1./(1+exp(-hidden_net));                          % use sigmoid activation function
                    hidden_act = this.hidden_act;
                    hidden_act_log(set,i,:) = this.hidden_act; 
                    
                    % calculate net input for output units
                    output_net_task = this.g(this.weights.W_TO) * transpose(task);      % input from task layer (task cue)
                    output_net_hidden = this.weights.W_HO * this.hidden_act;            % input from hidden layer
                    output_net_bias   = this.weights.W_BO * this.output_bias;           % input from output bias units
                    output_net = output_net_hidden + output_net_task + output_net_bias; % total net input to output layer
                    output_net_log(set,i,:) = tau*output_net + (1-tau) * squeeze(output_net_log(set,i-1,:)); % integrate input from previous time step and current time step
                    output_net = squeeze(output_net_log(set,i,:));
                    
                    % calculate activation of output units
                    this.output_act = 1./(1+exp(-output_net)); 
                    output_act = this.output_act;
                    output_act_log(set,i,:) = this.output_act'; 
                    
                    
                    % calculate MSE
                    if(~isempty(train))
                        MSE(set,i) = sum((output_act' - train).^2)./this.Noutput;
                    else
                        MSE(set,i) = 0; 
                    end
                
                end

            end
            
            % evidence accumulator 
            if(runAccumulator)
                  
                % drift, e.g. d = 0.1
                if(isfield(ddmp, 'd'))
                    d = ddmp.d;
                else
                    d = 0.1;
                end

                % threshold, e.g. z = 1
                if(isfield(ddmp, 'z'))
                    z = ddmp.z;
                else
                    z = 1;
                end

                % sigma, e.g. c = 0.1
                if(isfield(ddmp, 'c'))
                    c = ddmp.c;
                else
                    c = 0.1;
                end

                % wait time until recording
                if(isfield(ddmp, 'runs'))
                    respOnset = ddmp.respOnset;
                else
                    respOnset = 2;
                end

                % number of simulations
                if(isfield(ddmp, 'runs'))
                    nSimulations = ddmp.nSimulations;
                else
                    nSimulations = 10;
                end
                
                % create evidence matrix
                [output_sorted output_sorted_idx] = sort(output_act_log,3,'descend');
                
                % currently largest output value
                output_max = squeeze(output_sorted(:,:,1));
                output_max_idx = squeeze(output_sorted_idx(:,:,1));
                
                % currently second largest output value
                output_max2 = squeeze(output_sorted(:,:,2));
                output_max_idx2 = squeeze(output_sorted_idx(:,:,2));
                
                % build evidence matrix
                E = output_act_log;
%                 for outputIdx = 1:size(E, 3)
%                    maxMask = output_max;
%                    maxMask(output_max_idx == outputIdx) = output_max2(output_max_idx == outputIdx);
%                    E(:,:,outputIdx) = E(:,:,outputIdx)  - maxMask;
%                 end
                
                % create accumulation matrix
                A = repmat(E, [ 1, 1, 1, nSimulations]);
                A = normrnd(A.*d, c, size(A));      % scale evidence by drift rate and add noise
                A = cumsum(A, 2);   % accumulate evidence over time
            
                % performance matrices for reaction time and error rate
                DDM_RTs = nan(Nsets, nSimulations, length(z));
                DDM_ERs = nan(Nsets, nSimulations, length(z));
                
                % if training data is provided figure out correct response
                % unit for each pattern
                if(~isempty(trainA) && ~isempty(trainB))
                    C = repmat(trainB, [1, 1, nSimulations]);
                end
                
                start_z = 1; % threshold to search from
                % look at each time point
                for t = respOnset:iterations 
                    
                    % get snapshot
                    S = squeeze(A(:,t,:,:));
                    
                    for z_idx = start_z:length(z)
                        
                        % mark accumulators that are past the current threshold
                        threshMask = S > z(z_idx);
                        S_thresholded = squeeze(sum(threshMask,2)) >= 1;
                        
                        % calculate ER
                        accuracy = threshMask == C;
                        S_accuracy = squeeze(sum(accuracy, 2) == size(accuracy,2));
                        DDM_ERs_zIdx = DDM_ERs(:,:,z_idx);
                        pastThreshold = isnan(DDM_ERs(:,:,z_idx)) & S_thresholded;
                        DDM_ERs_zIdx(pastThreshold) = S_accuracy(pastThreshold);
                        DDM_ERs(:,:,z_idx) = DDM_ERs_zIdx;
                        
                        
                        % if accumulator hasn't passed the current threshold yet but just crossed threshold then assign time index as RT
                        DDM_RTs_zIdx = DDM_RTs(:,:,z_idx);
                        pastThreshold = isnan(DDM_RTs(:,:,z_idx)) & S_thresholded;
                        
                        
                        DDM_RTs_zIdx(pastThreshold & S_accuracy) = t;
                        DDM_RTs_zIdx(pastThreshold & (1-S_accuracy)) = -t;
                        DDM_RTs(:,:,z_idx) = DDM_RTs_zIdx;                    
                        

                        % if all accumulators passed the threshold, then remove threshold from list to improve performance
                        if(isequal(S_thresholded, ones(size(S_thresholded))))
                            start_z = z_idx + 1;
                        end
                        
                        
                    end
                    
                    
                end 
                
            end
            
        end
        
        % generate output given hidden layer activity and task input to
        % output layer
        function [output_act, hidden_act, MSE] = generateOutput(this, hidden_act, varargin)
            
            % parse arguments: task patterns
            if(length(varargin) >=1)
                task = varargin{1};
            else
                task = this.taskSet;
            end
            
            % calculate net input for output units
            output_net_task = this.g(this.weights.W_TO) * transpose(task);      % input from task layer (task cue)
            output_net_hidden = this.weights.W_HO * transpose(hidden_act);            % input from hidden layer
            output_net_bias   = this.weights.W_BO * this.output_bias;           % input from output bias units
            output_net = output_net_hidden + output_net_task + repmat(output_net_bias,1,size(output_net_hidden,2)); % total net input to output layer

            % calculate activation of output units
            output_act = 1./(1+exp(-output_net)); 

            % final network output
            output_act = output_act';
            
            % calculate MSE if correct output provided (train)
            MSE = -1;
            if(length(varargin) >=2)
                train = varargin{2};
                if(length(train) == length(output_act))
                    MSE = sum((output_act - train).^2)./this.Noutput;
                end
            end
            
        end
        
        % weight transformation function g(w) - used to calculate net input
        function weightsMod = g(this, weights)
            % weightsMod = abs(weights); % this weight transformation function ensures only positive weights
            weightsMod = weights;
        end
        
        % derivative of weight transformation function dg(w) - used to
        % calculate deltas for backpropagation
        function dWeights = dg(this, weights)
           %dWeights = sign(weights); % % this weight transformation function ensures only positive weights
           dWeights = 1;
        end
        
        function setData(this, inputData, taskData, trainData)
           this.inputSet = inputData;
           this.taskSet = taskData;
           this.trainSet = trainData;
        end

    end
    
end

