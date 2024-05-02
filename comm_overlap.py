from hta.trace_analysis import TraceAnalysis

# Load the Chrome trace file
path = "cs336-systems/cs336_systems/traces/naive"
analyzer = TraceAnalysis(trace_dir=path)
overlap_df = analyzer.get_comm_comp_overlap()