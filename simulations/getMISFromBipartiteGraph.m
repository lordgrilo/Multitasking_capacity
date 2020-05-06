function [maxCarryingCapacity, A_dual, pathwayCapacities,  BK_MIS] = getMISFromBipartiteGraph(A_bipartite)

% generate interference adjacency matrix from bipartite adjacency matrix
% code from Hasan K. Ozcimder & Biswadip Dey, 2015

A = A_bipartite;
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

% findMIS by  Roberto Olmi, 2010
% http://www.mathworks.com/matlabcentral/fileexchange/28470-heuristic-algorithm-for-finding-maximum-independent-set
MultiTask=findMIS(logical(A_dual),[1:m]);
BK_MIS = BK_MaxIS(logical(A_dual));
%
pathwayCapacities = [x y MultiTask];
% maxCarryingCapacity = sum(MultiTask);

% compute exact MIS code (code by Nesreen Ahmed at Intel Labs, 2017)
A = A_dual;
randomNumber = num2str(abs(round(randn*100)));
fileName = ['simmat_' randomNumber '.mat'];
save(fileName,'A'); %save to file
[~, maxCarryingCapacity] = system(['/home/musslick/.conda/envs/GioAnalysis/bin/python mis.py ' fileName]);
disp(maxCarryingCapacity);

end