#include "config.h"
#include "protocol_executer.hpp"

int main(int argc, char* argv[])
{
#if PROCESS_NUM > 1
    pid_t child_pid, wpid;
    int status = 0;
    for (int id = 0; id < PROCESS_NUM; id++)
    {
        if ((child_pid = fork()) == 0)
        {
            base_port +=
                (num_players * (num_players - 1) * id) + SPLIT_ROLES_OFFSET * num_players * (num_players - 1) *
                                                             PROCESS_NUM;  // offsets the starting port for each process
            process_offset =
                ((base_port - BASE_PORT) /
                 (num_players *
                  (num_players -
                   1)));  // offsets the starting input for each process, base port must be multiple of 1000 to work
            executeProgram(argc, argv, child_pid, PROCESS_NUM);  // child code
            exit(0);
        }
    }
    while ((wpid = wait(&status)) > 0)
        ;  // wait for all the child processes
#else
    base_port += SPLIT_ROLES_OFFSET * num_players * (num_players - 1);  // offsets the starting port for each process
    process_offset =
        ((base_port - BASE_PORT) /
         (num_players *
          (num_players -
           1)));  // offsets the starting input for each process, base port must be multiple of 1000 to work
    executeProgram(argc, argv, 0, 1);  // child code
#endif
    return 0;
}
