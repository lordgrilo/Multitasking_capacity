function A_dual = getDependencyGraph(A)

    % compute dependency graph
    %
    m = nnz(A);
    [x y] = find(A);

    % make sure that x,y are correctly aligned
    if(size(x,1) < size(x,2))
       x = x'; 
    end
    if(size(y,1) < size(y,2))
       y = y'; 
    end
    %
    A_dual = zeros(m);
    %
    for i=1:m
        for j = i+1:m
            first_in   = x(i);
            first_out  = y(i);
            second_in  = x(j);
            second_out = y(j);
            %
            if (first_in==second_in)
                A_dual(i,j) = 1;
            elseif (first_out==second_out)
                A_dual(i,j) = 1;
            elseif ~(isempty(find(y(find(x==first_in))==second_out))&&isempty(find(y(find(x==second_in))==first_out)))
                A_dual(i,j) = 1;
            end 
        end
    end
    %
    A_dual = (A_dual+A_dual');

end