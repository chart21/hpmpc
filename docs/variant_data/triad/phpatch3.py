# temporary (node-side only): absolute phase timestamps (CLOCK_MONOTONIC) of the preprocessing at the round-6 code
import sys
root = sys.argv[1] if len(sys.argv) > 1 else "/root/hpmpc"
T = lambda tag: f'{{ timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t); printf("PHASE {tag}: %.6f\\n", _t.tv_sec + _t.tv_nsec / 1e9); fflush(stdout); }}\n'
p = root + "/protocol_executer.hpp"; s = open(p).read()
reps = [
 ("        clock_gettime(CLOCK_REALTIME, &p1);\n        std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();\n#if MASK_FORWARD_ACTIVE\n",
  "        clock_gettime(CLOCK_REALTIME, &p1);\n        std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();\n        " + T("pre_start") + "#if MASK_FORWARD_ACTIVE\n"),
 ("            conv_early_start(ips, base_port, process_offset);\n            run_ot_phase(ips);\n",
  "            " + T("ot_start") + "            conv_early_start(ips, base_port, process_offset);\n            run_ot_phase(ips);\n            " + T("ot_end")),
 ("#else\n        run_ot_phase(ips);\n#endif\n#endif\n",
  "#else\n        " + T("ot_start") + "        run_ot_phase(ips);\n        " + T("ot_end") + "#endif\n#endif\n"),
 ("    RESULTTYPE garbage_PRE;\n", "    " + T("pass_start") + "    RESULTTYPE garbage_PRE;\n"),
 ("    FUNCTION<PROTOCOL_PRE<DATATYPE>>(&garbage_PRE);\n", "    FUNCTION<PROTOCOL_PRE<DATATYPE>>(&garbage_PRE);\n    " + T("pass_end")),
 ("    Iface::printTripleStats(CHEETAH_PARTY, process_offset);\n", "    " + T("complete_end") + "    Iface::printTripleStats(CHEETAH_PARTY, process_offset);\n"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w").write(s)
p = root + "/protocols/2-PC/aby2/aby2_pre.hpp"; s = open(p).read()
a = "        conv_async::join();  // the conv triples of the preprocessing pass, before the next generator takes the channels\n"
assert s.count(a) == 1
s = s.replace(a, a + "        " + T("conv_joined"))
open(p, "w").write(s)
print("patched")
