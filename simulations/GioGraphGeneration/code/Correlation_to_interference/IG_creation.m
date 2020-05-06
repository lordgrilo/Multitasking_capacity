%compute the interference graph using the weighted Jaccard of the
%activation layers

function IG_graph=IG_creation(dHidden, dOutput)
    %%% The input should be a tensor with dimensions:
    %%% T = # of tasks
    %%% I = # of input vectors
    %%% N = dimension of the hidden layer (space)
    %%% M = dimension of the output layer (space)
    
    [T, I, N] = size(dHidden); 
    [T, I, M] = size(dOutput);
    
    %%% This bit calculates the similarity matrices between hidden/output
    %%% layer activations corresponding to the same input. 
    O = zeros(T);
    H = zeros(T);

    for i=1:(T-1)
        for j=i+1:T
            for l=1:I
            O(i,j) = O(i,j) + sum(min(dOutput(i,l,:),dOutput(j,l,:)))/sum(max(dOutput(i,l,:),dOutput(j,l,:)));
            H(i,j) = H(i,j) + sum(min(dHidden(i,l,:),dHidden(j,l,:)))/sum(max(dHidden(i,l,:),dHidden(j,l,:)));
            end
        end
    end
    
    %%% This bit trivially averages the values obtained over the number of
    %%% inputs vectors
    
    Similarity_matrix_output = (O+O') / I
    Similarity_matrix_hidden = (H+H') / I
    
    %%% here I'm relying on the the definition of weights as the standard
    %%% weighted Jaccard index of the activation vectors relative to task t 
    %%% and task tt, averaged over the input space
    
    IG_graph = Similarity_matrix_output + Similarity_matrix_hidden;

    
    %%% the second bit here is the one that calculates the third interfer
    %%% as the over the rows of the similarity matrix having removed t and
    %%% tt task. 
    for t=1:T
        for tt=1:T
            if t~=tt
                v = Similarity_matrix_hidden(t,:);
                w = Similarity_matrix_output(tt,:);
                v([t,tt]) = [];
                w([t,tt]) = [];
                size(v), size(w)
                jacc = sum(min(v,w))/sum(max(v,w));
                IG_graph(t,tt) = IG_graph(t,tt) + jacc;
                IG_graph(tt,t) = IG_graph(tt,t) + jacc;
            end
        end
    end
    %%% Do we need a different normalization here? 
    





