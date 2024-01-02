#include "config.h"
#include "networking/buffers.h"
#include "protocol_executer.hpp"
#include <sys/types.h>
#include <sys/wait.h>

int main(int argc, char *argv[])
{
#if PROCESS_NUM > 1


pid_t child_pid, wpid;
int status = 0;

//Father code (before child processes start)

for (int id=0; id<PROCESS_NUM; id++) {
    if ((child_pid = fork()) == 0) {
        base_port += (num_players*(num_players-1)*id)+SPLIT_ROLES_OFFSET * num_players*(num_players-1) * PROCESS_NUM; //offsets the starting port for each process
        process_offset = ( (base_port - BASE_PORT) / (num_players*(num_players-1)) ); //offsets the starting input for each process, base port must be multiple of 1000 to work
        executeProgram(argc, argv, child_pid, PROCESS_NUM); //child code
        exit(0);
    }
}


while ((wpid = wait(&status)) > 0); // this way, the father waits for all the child processes 

//Father code (After all child processes end)
#else
    process_offset = ( (base_port % 1000) / (num_players*(num_players-1)) ); //offsets the starting input for each process, base port must be multiple of 1000 to work
    executeProgram(argc, argv, 0, 1); //child code
#endif
return 0;
}
