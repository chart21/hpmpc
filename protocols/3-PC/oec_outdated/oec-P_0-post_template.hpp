#pragma once
#include "../sharemind/sharemind_base.hpp"
#define SHARE DATATYPE
class OEC0_POST
{
bool optimized_sharing;
public:
OEC0_POST(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

XOR_Share public_val(DATATYPE a)
{
    return a;
}

DATATYPE Not(DATATYPE a)
{
   return a;
}

template <typename func_add>
DATATYPE Add(DATATYPE a, DATATYPE b, func_add ADD)
{
   return a;
}


template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(DATATYPE a, DATATYPE b, DATATYPE &c, func_add ADD, func_sub SUB, func_mul MULT)
{
}


template <typename func_add, typename func_sub>
void complete_mult(DATATYPE &c, func_add ADD, func_sub SUB)
{
}

void prepare_reveal_to_all(DATATYPE a)
{
}    



template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(DATATYPE a, func_add ADD, func_sub SUB)
{
a = XOR(retrieve_output_share(),receive_from_live(P_2));
return a;
}

XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


#if OPT_SHARE == 0 && PREP_SHARE == 0
void share_unoptimized(DATATYPE a[], int id, int l)
{

if(id == P_0)
{
    for(int i = 0; i < l; i++)
    {
    //special sharing technique, P_0 keeps it inputs, the other parties hold share=0
    a[i] = get_input_live();
    /* sending_args[0].sent_elements[sending_rounds][sb] = 0; */
    /* sending_args[1].sent_elements[sending_rounds][sb] = 0; */
    /* sb += 1; */
    DATATYPE r = getRandomVal(P_0); //should be an SRNG shared by P_0,P_1,P_2 to save communication
    a[i] = XOR(r,a[i]);
    send_to_live(P_1, a[i]);
    send_to_live(P_2, a[i]);
    a[i] = r;
    }

}
}
#endif

template <typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P_0)
{
    #if OPT_SHARE == 0 && PREP_SHARE == 0
    share_unoptimized(a, id, l);
    #else
    for(int i = 0; i < l; i++)
    {
    //special sharing technique, P_0 keeps it inputs, the other parties hold share=0
    a[i] = get_input_live();
        /* sending_args[0].sent_elements[sending_rounds][sb] = 0; */
    /* sending_args[1].sent_elements[sending_rounds][sb] = 0; */
    /* sb += 1; */
    /* DATATYPE r = getRandomVal(2); //should be an SRNG shared by P_0,P_1,P_2 to save communication */
    /* a[i] = XOR(r,a[i]); */
    /* sending_args[0].sent_elements[sending_rounds][sb] = SET_ALL_ZERO(); */
    /* sending_args[1].sent_elements[sending_rounds][sb] = SET_ALL_ZERO(); */
    /* sb += 1; */
    /* a[i] = r; */
    }
#endif 
}
else{
for(int i = 0; i < l; i++)
    {
    a[i] = getRandomVal(id);
    }
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
    return;
}





void send()
{
    send_live();
}

void receive()
{
    receive_live();
}

void communicate()
{
    communicate_live();
}
};
