function A = getGraphFromEdgeList(filepath)

    fid = fopen(filepath);
    
    inputNodes  = [];
    outputNodes = [];

    % read first line
    tline = fgets(fid);
    
    % loop through whole file
    while ischar(tline)
        
        % initialize input/output node
        inputNode = nan;
        outputNode = nan;
        
        % get separators
        sepIdx = strfind(tline,' ');
        
        % read out input node
        inputNode = str2double(tline(1:(sepIdx(1)-1)));
        
        % read output node
        if(length(sepIdx) >= 2)
            outputNode = str2double(tline( (sepIdx(1)+1) : (sepIdx(2)-1) ));
        else
            outputNode = str2double(tline( (sepIdx(1)+1) : end ));
        end
        
        % if valid numbers were obtained, add them to the list
        if(~isnan(inputNode) && ~isnan(outputNode) && isnumeric(inputNode) && isnumeric(outputNode))
            inputNodes = [inputNodes; inputNode];
            outputNodes = [outputNodes; outputNode];
        end
        
        % read new line
        tline = fgets(fid);
    end
    
    % close file
    fclose(fid);
    
    % offset input nodes
    inputNodes = inputNodes + 1;
    outputNodes = outputNodes + 1;
    
    % determine number of input nodes
    dimInput = max(inputNodes);
    
    % offset output node idx
    outputNodes = outputNodes - dimInput;

    % determine number of output nodes
    dimOutput = max(outputNodes);
    
    % create graph
    A = zeros(dimInput, dimOutput);
    
    % fill with edges
    for edgeIdx = 1:length(inputNodes)
        A(inputNodes(edgeIdx), outputNodes(edgeIdx)) = 1;
    end
    
    
end