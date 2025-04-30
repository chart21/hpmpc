// TODO: change to vector or make somehow usable for more than 3 players
#define RR                 \
    int send_rounds[2] = { \
        8,                 \
        8,                 \
    };
#define SR                \
    int rec_rounds[2] = { \
        8,                \
        8,                \
    };
#define ETR                       \
    int elements_to_rec[2][8] = { \
        {                         \
            262144,               \
            0,                    \
            0,                    \
            0,                    \
            0,                    \
            0,                    \
            0,                    \
            0,                    \
        },                        \
        {                         \
            64,                   \
            0,                    \
            0,                    \
            0,                    \
            0,                    \
            0,                    \
            0,                    \
            0,                    \
        },                        \
    };
#define ETS                        \
    int elements_to_send[2][8] = { \
        {                          \
            0,                     \
            0,                     \
            0,                     \
            0,                     \
            0,                     \
            0,                     \
            0,                     \
            1,                     \
        },                         \
        {                          \
            0,                     \
            0,                     \
            0,                     \
            0,                     \
            0,                     \
            0,                     \
            0,                     \
            1,                     \
        },                         \
    };
