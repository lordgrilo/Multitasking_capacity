function [tasks MaxMem] = GenMultiTasks(N, activeTasks)

    MaxMem = 0;
    
    Nactive = size(activeTasks, 2);
    NtaskSets = size(activeTasks, 1);
    
    relevantTasks = unique(activeTasks);
    
    if(NtaskSets == 0 || Nactive+1 > length(relevantTasks))
        tasks = [];
        return
    end
    
    % Pre-allocate the memory for the available task sets. This is a worst
    % case scenario where all tasks are independent, which means any
    % combination of them is valid.
    %fprintf(1, '#RelTasks=%d, Nactive+1=%d\n', length(relevantTasks), Nactive+1); 
    tasks = zeros(nchoosek(length(relevantTasks), Nactive+1), Nactive+1);
    taskSetIdx = 1;
    for ii=1:NtaskSets
    
        % Define a completely full adjacency matrix, all tasks specified.
        A = ones(N);
        
        % Now, only mark relevant tasks as available by setting them to 0
        A(relevantTasks) = 0;
        
        % We are using column major for the task indices, this is messy but
        % that is how the 
        A = A';
        
        % Lets convert our active tasks to 2D indexes. This will help us
        % know on which rows and columns in the adjacency matrix they reside.
        %[trows, tcols] = ind2sub([N N], activeTasks(ii,:));
        tcols = rem(activeTasks(ii,:)-1,N)+1;
        trows = (activeTasks(ii,:)-tcols)/N + 1;
        
        % Now, specify the active tasks. In addition, mark any task that
        % shares a row or column with the task as un-available (set it to 1 too)
        A(trows, :) = 1;
        A(:, tcols) = 1;
        
        % Find all linear indices that are still available.
        A = A';
        idx = relevantTasks(A(relevantTasks) ~= 1);
        %idx = find(A' ~= 1);

        % Now, remove any task that is less than the maximum current active
        % task. We do this because it ensures that we don't enumerate 
        % duplicate task patterns.
        maxActiveTask = activeTasks(ii, end);
        idx = idx(idx > maxActiveTask);
        
%         maxActiveTask = max(activeTasks(ii, :));
%         maxIdx = 1;
%         while(idx(maxIdx) < maxActiveTask & maxIdx < length(idx))
%             maxIdx = maxIdx + 1;
%         end
%         idx = idx(maxIdx:end);
%         
        
        % Add the task set to our list
        zz = 1;
        for kk=taskSetIdx:(taskSetIdx+length(idx)-1)
            tasks(kk, :) = [activeTasks(ii, :) idx(zz)];
            zz = zz + 1;
        end
        %tasks(taskSetIdx:taskSetIdx+length(idx)-1,:) = [repmat(activeTasks(ii, :), [length(idx) 1]) idx];
        taskSetIdx = taskSetIdx + length(idx);        
    end
  
    M = whos('tasks');
    bytes = max([M.bytes]);
         
    if(MaxMem < bytes)
        MaxMem = bytes;
    end
    
    % We allocated too much space, remove any unused rows
    tasks(all(tasks == 0, 2), :) = [];
    
    % There will also be some duplicate tasks sets, remove them.
%    tasks = sort(tasks, 2);
%    tasks = sortrows(tasks);
%    tasks = unique(tasks, 'rows');
   
end