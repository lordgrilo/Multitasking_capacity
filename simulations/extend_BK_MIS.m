function full_BK_MIS = extend_BK_MIS(BK_MIS)

    full_BK_MIS = [];
    
    for col = 1:size(BK_MIS,2)
        
        full_BK_MIS = [full_BK_MIS BK_MIS(:,col)];
        
        numTasks = sum(BK_MIS(:,col));
        tasksIdx = find(BK_MIS(:,col) == 1);
        
        for capacity = 2:(sum(BK_MIS(:,col))-1);
            
            taskCombs = 1:numTasks;
            for i = 2:capacity
               taskCombs = combvec(taskCombs, 1:numTasks); 
                   for j = 1:(i-1)
                        idicies = find(taskCombs(j,:) == taskCombs(i,:) & taskCombs(j,:) ~= 0);
                        taskCombs(:,idicies) = [];
                   end
            end
           
            taskCombs = sort(taskCombs,1);
            taskCombs = transpose(unique(transpose(taskCombs),'rows'));
            
            for i = 1:size(taskCombs,2)
                full_BK_MIS = [full_BK_MIS zeros(size(full_BK_MIS,1),1)];
                full_BK_MIS(tasksIdx(taskCombs(:,i)),end) = 1;
            end
            
        end
        
    end
        
    % remove duplicates
    full_BK_MIS = transpose(unique(transpose(full_BK_MIS), 'rows'));
    
end