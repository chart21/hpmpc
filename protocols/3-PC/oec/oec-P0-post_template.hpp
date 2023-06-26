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

// Receive sharing of ~XOR(a,b) locally
DATATYPE Xor(DATATYPE a, DATATYPE b)
{
   return a;
}



//prepare AND -> send real value a&b to other P
void prepare_and(DATATYPE a, DATATYPE b, DATATYPE &c)
{
}


void complete_and(DATATYPE &c)
{
}

void prepare_reveal_to_all(DATATYPE a)
{
}    



DATATYPE complete_Reveal(DATATYPE a)
{
a = XOR(a,receive_from_live(P2));
return a;
}

XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


#if OPT_SHARE == 0 && PREP_SHARE == 0
void share_unoptimized(DATATYPE a[], int id, int l)
{

if(id == P0)
{
    for(int i = 0; i < l; i++)
    {
    //special sharing technique, P0 keeps it inputs, the other parties hold share=0
    a[i] = get_input_live();
    /* sending_args[0].sent_elements[sending_rounds][sb] = 0; */
    /* sending_args[1].sent_elements[sending_rounds][sb] = 0; */
    /* sb += 1; */
    DATATYPE r = getRandomVal(P0); //should be an SRNG shared by P0,P1,P2 to save communication
    a[i] = XOR(r,a[i]);
    send_to_live(P1, a[i]);
    send_to_live(P2, a[i]);
    a[i] = r;
    }

}
}
#endif

void prepare_receive_from(DATATYPE a[], int id, int l)
{
if(id == P0)
{
    #if OPT_SHARE == 0 && PREP_SHARE == 0
    share_unoptimized(a, id, l);
    #else
    for(int i = 0; i < l; i++)
    {
    //special sharing technique, P0 keeps it inputs, the other parties hold share=0
    a[i] = get_input_live();
        /* sending_args[0].sent_elements[sending_rounds][sb] = 0; */
    /* sending_args[1].sent_elements[sending_rounds][sb] = 0; */
    /* sb += 1; */
    /* DATATYPE r = getRandomVal(2); //should be an SRNG shared by P0,P1,P2 to save communication */
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

// party will not receive output
void complete_receive_from(DATATYPE a[], int id, int l)
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
