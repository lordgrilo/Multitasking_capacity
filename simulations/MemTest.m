function [] = MemTest()

    function [taskCombs MaxMem] = BadGen(NPathways)
        MaxMem = 0;
        
        for Nactive = 2:NPathways

            taskPaths = [zeros(1,NPathways-Nactive) 1:NPathways];
            taskCombs = taskPaths;

            for k = 2:1:NPathways
                taskCombs = combvec(taskCombs,taskPaths); % 1:NPathways

                M = whos('taskCombs');
                bytes = max([M.bytes]);
                
                if(MaxMem < bytes)
                    MaxMem = bytes;
                end
                
                for j = 1:(k-1)
                    indicies = find(taskCombs(j,:) == taskCombs(k,:) & taskCombs(j,:) ~= 0);
                    taskCombs(:,indicies) = [];
                end
            end  

        end
    end

    function [myTasks MaxMem] = GoodGen(NPathways, relevantTasks)
        myTasks = relevantTasks;

        MaxMem = 0;
        
        for Nactive = 2:NPathways
            [myTasks Mem] = GenMultiTasks(NPathways, myTasks); 
            if(Mem > MaxMem)
                MaxMem = Mem;
            end
        end
    end

    NRange = 2:7;

    for ii=NRange
        [taskCombs BadMemUsed] = BadGen(ii);
        BadMem(ii-1) = BadMemUsed / 1024^2;
    end

    NRange2 = 2:25;
    for ii=NRange2
        ii
        [taskCombs GoodMemUsed] = GoodGen(ii, find(eye(ii)));
        GoodMem(ii-1) = GoodMemUsed / 1024^2;
    end
    
    size(taskCombs)
    
    plot(NRange, BadMem, '-r', 'DisplayName', 'Old Algorithm');
    hold on;
    plot(NRange2, GoodMem, '-b', 'DisplayName', 'New Algorithm');
    xlabel('Graph Size');
    ylabel('Max Memory Used (MB)');
    
end