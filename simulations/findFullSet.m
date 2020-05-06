% author: Sebastian Musslick

function [set] = findFullSet(A, row, thresh, currSet) 

    set = [];
    
    for i = 1:size(A,2)
       
        if(A(row, i) > thresh && ~ismember(i, currSet))

            set = [set i];
            set = [set findFullSet(A, i, thresh, [currSet set])];
            currSet = [currSet set];
            
        end

    end

end