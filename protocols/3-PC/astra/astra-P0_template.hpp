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

// Receive sharing of ~XOR(a,b) locally
Coordinator_Share Xor(Coordinator_Share a, Coordinator_Share b)
{
   return XOR(a,b);
}



//prepare AND -> send real value a&b to other P
void prepare_and(Coordinator_Share a, Coordinator_Share b, Coordinator_Share &c)
{
DATATYPE yxy = AND(a,b);
c = XOR(getRandomVal(P1),getRandomVal(P2)); //yz
DATATYPE yxy2 = XOR(getRandomVal(P1),yxy);
send_to_live(P2, yxy2);
}

void complete_and(Coordinator_Share &c)
{
}

void prepare_reveal_to_all(Coordinator_Share a)
{
    send_to_live(P1, a);
    send_to_live(P2, a);
}    



DATATYPE complete_Reveal(Coordinator_Share a)
{
return XOR(a,receive_from_live(P2));
}


Coordinator_Share* alloc_Share(int l)
{
    return new Coordinator_Share[l];
}



void prepare_receive_from(Coordinator_Share a[], int id, int l)
{
if(id == P0)
{
    if(optimized_sharing == true)
    {
    for(int i = 0; i < l; i++)
    {
    a[i] = get_input_live(); 
    DATATYPE lx1 = getRandomVal(P1);
    send_to_live(P2, XOR(a[i],lx1));
    }

    }
    else
    {
    for(int i = 0; i < l; i++)
    {
    DATATYPE lv1 = getRandomVal(P1); 
    DATATYPE lv2 = getRandomVal(P2);
    a[i] = XOR(lv1,lv2);// lv
    DATATYPE mv = XOR(a[i],get_input_live());
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

void complete_receive_from(Coordinator_Share a[], int id, int l)
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
