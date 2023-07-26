#pragma once
#include "astra_base.hpp"
class ASTRA0
{
bool optimized_sharing;
public:
ASTRA0(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

Coordinator_Share public_val(DATATYPE a)
{
    return SET_ALL_ZERO();
}

Coordinator_Share Not(Coordinator_Share a)
{
   return a;
}

template <typename func_add>
Coordinator_Share Add(Coordinator_Share a, Coordinator_Share b, func_add ADD)
{
   return ADD(a,b);
}



template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(Coordinator_Share a, Coordinator_Share b, Coordinator_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
DATATYPE yxy = MULT(a,b);
c = ADD(getRandomVal(P1),getRandomVal(P2)); //yz
DATATYPE yxy2 = SUB(yxy,getRandomVal(P1)); //yxy,2
send_to_live(P2, yxy2);
}

template <typename func_add, typename func_sub>
void complete_mult(Coordinator_Share &c, func_add ADD, func_sub SUB)
{
}

void prepare_reveal_to_all(Coordinator_Share a)
{
    send_to_live(P1, a);
    send_to_live(P2, a);
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(Coordinator_Share a, func_add ADD, func_sub SUB)
{
return SUB(receive_from_live(P2),a);
}


Coordinator_Share* alloc_Share(int l)
{
    return new Coordinator_Share[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(Coordinator_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P0)
{
    if(optimized_sharing == true)
    {
    for(int i = 0; i < l; i++)
    {
    a[i] = get_input_live(); 
    DATATYPE lx1 = getRandomVal(P1);
    send_to_live(P2, ADD(a[i],lx1));
    }

    }
    else
    {
    for(int i = 0; i < l; i++)
    {
    DATATYPE lv1 = getRandomVal(P1); 
    DATATYPE lv2 = getRandomVal(P2);
    a[i] = ADD(lv1,lv2);// lv
    DATATYPE mv = ADD(a[i],get_input_live());
    send_to_live(P1, mv);
    send_to_live(P2, mv);
    }

    }
}
else if(id == P1){
for(int i = 0; i < l; i++)
    {
    a[i] = getRandomVal(P1);
    
    }


}
else if(id == P2)// id ==2
{
    for(int i = 0; i < l; i++)
    {
        a[i] = getRandomVal(P2);
    }

}
}

template <typename func_add, typename func_sub>
void complete_receive_from(Coordinator_Share a[], int id, int l, func_add ADD, func_sub SUB)
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
