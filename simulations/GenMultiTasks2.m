function tasks = GenMultiTasks2(NPathways, Nactive, relevantTasks)

    relevantTaskM = zeros(NPathways, NPathways);
    relevantTaskM(relevantTasks) = 1;
    relevantTaskM = transpose(relevantTaskM); % input components are rows, output components are columns

    taskPaths = [zeros(1,NPathways-Nactive) 1:NPathways];
    taskCombs = taskPaths;

    for k = 2:1:NPathways
        taskCombs = combvec(taskCombs,taskPaths); % 1:NPathways
        for j = 1:(k-1)
            indicies = find(taskCombs(j,:) == taskCombs(k,:) & taskCombs(j,:) ~= 0);
            taskCombs(:,indicies) = [];
        end
    end

    % delete multitasking conditions that exceed specified number of active
    % tasks
    numNonZeroElements = taskCombs ~= 0;
    numNonZeroElements = sum(numNonZeroElements);
    taskCombs(:,numNonZeroElements ~= Nactive) = [];
    taskCombs = transpose(unique(transpose(taskCombs),'rows'));

    % I want to sort the tasks by row then column. They are encoded so that
    % values of 0 correspond to empty rows. I make these a large value so 
    % that they will be sorted last in the order of combinations. 
    taskCombs(taskCombs == 0) = 1e38;
    taskCombs = sortrows(taskCombs', 1:size(taskCombs,1))';
    taskCombs(taskCombs == 1e38) = 0;
    
    validTasks = logical(zeros(size(taskCombs,2),1));
    for currTaskComb = 1:size(taskCombs,2)

        % build task input
        currTasksM = zeros(NPathways, NPathways); % input components are rows, output components are columns

        for k = 1:NPathways;
            if(taskCombs(k,currTaskComb) ~= 0)
                currTasksM(k,taskCombs(k,currTaskComb)) = 1;
            end
        end
    
        if(max(sum(currTasksM,1)) > 1) % hard constraint
            error('Overlapping tasks: Can''t use one output modality for two different feature dimensions');
        end
        if(max(sum(currTasksM,2)) > 1) % soft constraint (can potentially be removed)
            error('Overlapping feature dimensions: Can''t perform two tasks on the same input features');
        end
        
        % check if task combinations only contains relevant tasks
        if( mean(mean((relevantTaskM & currTasksM) == currTasksM)) == 1)
            validTasks(currTaskComb) = 1;
        end
    end
    taskCombs = taskCombs(:, validTasks);
    
    % Go through each task and convert to row-major
    tasks = zeros(size(taskCombs,2), Nactive);
    for ii=1:size(taskCombs,2)
        % build task input
        currTasksM = zeros(NPathways, NPathways); % input components are rows, output components are columns

        for k = 1:NPathways;
            if(taskCombs(k,ii) ~= 0)
                currTasksM(k,taskCombs(k,ii)) = 1;
            end
        end
        
        tasks(ii, :) = find(currTasksM')';
    end
    
    %tasks = sortrows(tasks, 1:size(tasks,2));
    
end