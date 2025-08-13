from argparse import ArgumentParser, FileType

from flow_analysis.readers import readers
from flow_analysis.measurements.scales import measure_w0
from flow_analysis.measurements.Q import Q_mean
from flow_analysis.stats.autocorrelation import exp_autocorrelation_fit

from .provenance import get_basic_metadata, text_metadata


def get_args():
    parser = ArgumentParser(description="Compute basic observables from Wilson flow")
    parser.add_argument("flow_filename", help="Grid flow log to analyse")
    parser.add_argument(
        "--tag", default="", help="Tag to distinguish ensemble in combined CSVs"
    )
    parser.add_argument(
        "--output_file",
        type=FileType("w"),
        default="-",
        help="File in which to place output (default: stdout)",
    )
    parser.add_argument(
        "--W0", type=float, default=0.28125, help="Reference scale for W0 computation"
    )
    parser.add_argument(
        "--fileformat",
        default="grid",
        choices=["grid", "hirep"],
        help="File format logs are in",
    )
    return parser.parse_args()


def write_output(output_file, tag, w0, top_charge, tau_exp_Q):
    print(text_metadata(get_basic_metadata()), file=output_file)
    print("name,w_0,w_0_error,<Q>,<Q>_err,tau_Q,err_tau_Q", file=output_file)
    print(
        ",".join(
            map(
                str,
                [
                    tag,
                    w0.nominal_value,
                    w0.std_dev,
                    top_charge.nominal_value,
                    top_charge.std_dev,
                    tau_exp_Q.nominal_value,
                    tau_exp_Q.std_dev,
                ],
            )
        ),
        file=output_file,
    )


def main():
    args = get_args()
    flows = readers[args.fileformat](args.flow_filename, check_consistency=False)
    w0 = measure_w0(flows, args.W0, operator="plaq")
    top_charge = Q_mean(flows, w0.nominal_value**2)
    tau_exp_Q = exp_autocorrelation_fit(flows.Q_history(w0.nominal_value**2))
    write_output(args.output_file, args.tag, w0, top_charge, tau_exp_Q)


if __name__ == "__main__":
    main()
