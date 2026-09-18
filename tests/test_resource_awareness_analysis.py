from scripts.analyze_resource_awareness import summarize


def test_resource_summary_computes_mean_and_sample_std() -> None:
    records = []
    for heterogeneity in (1, 5, 10):
        for method in (
            "fixed_hierarchical", "resource_tree",
            "drift_no_resource_awareness", "drift",
        ):
            for seed, value in zip((42, 43, 44), (1.0, 2.0, 3.0)):
                row = {"heterogeneity": heterogeneity, "method": method, "seed": seed}
                row.update({
                    metric: value
                    for metric in (
                        "final_accuracy", "mean_round_latency_sim_ms",
                        "p95_round_latency_sim_ms", "max_capacity_normalized_load",
                        "capacity_normalized_load_imbalance", "max_peak_node_traffic_bytes",
                        "total_bytes", "total_messages", "aggregation_load_imbalance",
                    )
                })
                records.append(row)
    result = summarize(records)
    assert len(result) == 12
    assert result[0]["mean_round_latency_sim_ms_mean"] == 2.0
    assert result[0]["mean_round_latency_sim_ms_std"] == 1.0
