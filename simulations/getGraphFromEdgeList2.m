function A = getGraphFromEdgeList2(filepath)
    
    % Get the edge list from the file
    E = dlmread(filepath, ' ');
    
    A = zeros(10,10); 
    for ii=1:size(E,1) 
        A(E(ii,1)+1, E(ii,2)+1) = 1;
        A(E(ii,2)+1, E(ii,1)+1) = 1;
    end
    
    %A = A(sum(A,2) > 0, sum(A,1) > 0);
    
end