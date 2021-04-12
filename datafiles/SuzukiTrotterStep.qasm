OPENQASM 2.0;
include "qelib1.inc";
gate circuit-11(param0,param1,param2,param3) q0,q1 { circuit-12(0.0170000000000000,0.0170000000000000,0.0170000000000000) q0,q1; circuit-18(0.0294448637286709) q0,q1; }
qreg s[2];
creg b[2];
circuit-11(0.0294448637286709,0.0170000000000000,0.0170000000000000,0.0170000000000000) s[0],s[1];
measure s[0] -> b[0];
measure s[1] -> b[1];
