#pragma once
template <typename Datatype>
class Sharemind_init
{
  public:
    Sharemind_init() {}

    template <typename func_add>
    Sharemind_init Add(const Sharemind_init b, func_add ADD) const
    {
        return Sharemind_init();
    }

    Sharemind_init public_val(const Datatype a) { return Sharemind_init(); }

    Sharemind_init Not() const { return Sharemind_init(); }

    template <typename func_add, typename func_sub, typename func_mul>
    Sharemind_init prepare_mult(const Sharemind_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        send_to_(PNEXT);
        send_to_(PPREV);
        return Sharemind_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void complete_mult(func_add ADD, func_sub SUB, func_mul MULT)
    {
        receive_from_(PNEXT);
        receive_from_(PPREV);
    }

    void prepare_reveal_to_all() const
    {
        for (int t = 0; t < num_players - 1; t++)
            send_to_(t);
    }

    template <typename func_mul>
    Sharemind_init mult_public(const Datatype b, func_mul MULT) const
    {
        return Sharemind_init();
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        for (int t = 0; t < num_players - 1; t++)
            receive_from_(t);
        return SET_ALL_ZERO();
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
    }

    static void send() { send_(); }
    static void receive() { receive_(); }
    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }
};
