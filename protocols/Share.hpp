#pragma once

template <typename Datatype, typename Share_Type>
class Share
{
    private:
    Share_Type& share()
    {
        return static_cast<Share_Type&>(*this);
    }
    public:
Share_Type Not(Share a)
{
   return share().Not(a);
}

template <typename func_add>
Share_Type Add(Share_Type a, Share_Type b, func_add ADD)
{
    return share().Add(a,b,ADD);
}

template <typename func_add, typename func_sub, typename func_mul>
    Share_Type prepare_mult(Share_Type b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return share().prepare_mult(b,ADD,SUB,MULT);
}

void prepare_reveal_to_all()
{
    share().prepare_reveal_to_all();
}    

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
    return share().complete_Reveal(ADD,SUB);
}

template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
    share().template prepare_receive_from<id>(ADD,SUB);
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    share().template complete_receive_from<id>(ADD,SUB);
}



static void send()
{
    Share_Type::send();
}

static void receive()
{
    Share_Type::receive();
}

static void communicate()
{
    Share_Type::communicate();
}
        

};
