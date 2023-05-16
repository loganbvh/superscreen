import argparse

import superscreen as sc

from . import huber, hypres, ibm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of solver iterations.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=100,
        help="Number of Laplacian mesh smoothing steps to perform.",
    )
    parser.add_argument(
        "--no-terminals", action="store_true", help="Set with_terminals=False"
    )

    args = parser.parse_args()

    squid_funcs = {
        "hypres-small": hypres.small.make_squid,
        "ibm-small": ibm.small.make_squid,
        "ibm-medium": ibm.medium.make_squid,
        "ibm-large": ibm.large.make_squid,
        "ibm-xlarge": ibm.xlarge.make_squid,
        "huber": huber.make_squid,
    }

    max_edge_lengths = {
        "hypres-small": 0.2,
        "ibm-small": 0.1,
        "ibm-medium": 0.1,
        "ibm-large": 0.15,
        "ibm-xlarge": 0.4,
        "huber": 0.4,
    }

    mutuals = {}
    for name, make_squid in squid_funcs.items():
        squid: sc.Device = make_squid(with_terminals=(not args.no_terminals))
        squid.make_mesh(
            max_edge_length=max_edge_lengths[name],
            smooth=args.smooth,
        )
        if args.no_terminals:
            M = squid.mutual_inductance_matrix(
                iterations=args.iterations, units="Phi_0 / A"
            )
        else:
            I_fc = "1 mA"
            solution = sc.solve(
                squid,
                terminal_currents={"fc": {"source": I_fc, "drain": f"-{I_fc}"}},
                iterations=args.iterations,
            )[-1]
            M = sum(solution.hole_fluxoid("pl_center")) / sc.ureg(I_fc)
        M.ito("Phi_0 / A")
        mutuals[make_squid.__module__] = M.to("Phi_0 / A")
        print(f"{name!r}: {M:.3f~P}")

    for label, mutual in mutuals.items():
        print()
        print(label)
        print("-" * len(label))
        print(mutual)
        print(mutual.to("pH"))
        print("-" * len(str(mutual)))
