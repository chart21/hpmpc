// sealbench: per-operation costs of SEAL 4.1 BFV at N = 4096 (primes 60, 49) and N = 8192 (60, 49 + special 60):
// forward NTT of one polynomial over all data primes, plaintext x ciphertext multiply (NTT form), ciphertext add,
// and one Galois automorphism with key switching (N = 8192 only). Single thread, microseconds per operation.
#include <seal/seal.h>
#include <seal/util/ntt.h>
#include <chrono>
#include <cstdio>
#include <vector>
using namespace seal;
using clk = std::chrono::steady_clock;
template <class F> double us(int reps, F&& f) {
    auto t0 = clk::now();
    for (int i = 0; i < reps; i++) f();
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count() / reps;
}
void run(size_t N, std::vector<int> bits, bool keyswitch) {
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
    parms.set_plain_modulus(uint64_t(1) << 32);
    SEALContext ctx(parms, true, keyswitch ? sec_level_type::tc128 : sec_level_type::none);
    KeyGenerator kg(ctx);
    PublicKey pk; kg.create_public_key(pk);
    Encryptor enc(ctx, pk); Evaluator ev(ctx);
    Plaintext pt(N); for (size_t i = 0; i < N; i++) pt[i] = (i * 2654435761u) & 0xffffffff;
    Ciphertext ct; enc.encrypt(pt, ct);
    auto parms_id = ct.parms_id();
    Ciphertext ntt_ct = ct; ev.transform_to_ntt_inplace(ntt_ct);
    Plaintext ntt_pt = pt; ev.transform_to_ntt_inplace(ntt_pt, parms_id);
    auto cd = ctx.get_context_data(parms_id);
    auto tables = cd->small_ntt_tables();
    size_t L = cd->parms().coeff_modulus().size();
    std::vector<uint64_t> poly(N * L, 12345);
    double t_ntt = us(2000, [&] { for (size_t j = 0; j < L; j++) util::ntt_negacyclic_harvey(poly.data() + j * N, tables[j]); });
    Ciphertext acc = ntt_ct, tmp;
    double t_mul = us(2000, [&] { tmp = ntt_ct; ev.multiply_plain_inplace(tmp, ntt_pt); ev.add_inplace(acc, tmp); });
    double t_add = us(4000, [&] { ev.add_inplace(acc, ntt_ct); });
    printf("N=%zu data primes=%zu: NTT(one poly, all primes) %.1f us, mult_plain+add (ct) %.1f us, add %.1f us",
           N, L, t_ntt, t_mul, t_add);
    if (keyswitch) {
        GaloisKeys gk;
        std::vector<uint32_t> elts;
        for (uint32_t g = 3; g < 2 * N; g = 2 * g - 1) { elts.push_back(g); if (elts.size() >= 13) break; }
        kg.create_galois_keys(elts, gk);
        std::stringstream ss; auto sz = kg.create_galois_keys(elts).save(ss);
        Ciphertext c2 = ct;
        double t_gal = us(300, [&] { ev.apply_galois_inplace(c2, elts[0], gk); });
        printf(", apply_galois (keyswitch) %.1f us; %zu Galois keys serialized %.2f MiB", t_gal, elts.size(), sz / 1048576.0);
    }
    printf("\n");
}
int main() {
    run(4096, {60, 49}, false);
    run(8192, {60, 49, 60}, true);
}
