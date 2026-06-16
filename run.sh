python run.py train --episodes 2500 --device cuda --benchmark small \
--results-path outputs/small_legacy/training_results.json \
--checkpoint-path outputs/small_legacy/gppo_policy.pt \
--constraint-mode legacy

python run.py train --episodes 2500 --device cuda --benchmark small \
--results-path outputs/small_strict_connectivity_only/training_results.json \
--checkpoint-path outputs/small_strict_connectivity_only/gppo_policy.pt \
--constraint-mode strict_connectivity_only

python run.py train --episodes 2500 --device cuda --benchmark small \
--results-path outputs/small_strict_connectivity_plus_capacity/training_results.json \
--checkpoint-path outputs/small_strict_connectivity_plus_capacity/gppo_policy.pt \
--constraint-mode strict_connectivity_plus_capacity

python run.py train --episodes 2500 --device cuda --benchmark small \
--results-path outputs/small_strict_connectivity_plus_capacity_plus_bandwidth/training_results.json \
--checkpoint-path outputs/small_strict_connectivity_plus_capacity_plus_bandwidth/gppo_policy.pt \
--constraint-mode strict_connectivity_plus_capacity_plus_bandwidth

python run.py train --episodes 2500 --device cuda --benchmark small \
--results-path outputs/small_strict_full/training_results.json \
--checkpoint-path outputs/small_strict_full/gppo_policy.pt \
--constraint-mode strict_full

python run.py visualize --results-path outputs/small_legacy/training_results.json --checkpoint-path outputs/small_legacy/gppo_policy.pt
python run.py visualize --results-path outputs/small_strict_connectivity_only/training_results.json --checkpoint-path outputs/small_strict_connectivity_only/gppo_policy.pt
python run.py visualize --results-path outputs/small_strict_connectivity_plus_capacity/training_results.json --checkpoint-path outputs/small_strict_connectivity_plus_capacity/gppo_policy.pt
python run.py visualize --results-path outputs/small_strict_connectivity_plus_capacity_plus_bandwidth/training_results.json --checkpoint-path outputs/small_strict_connectivity_plus_capacity_plus_bandwidth/gppo_policy.pt
python run.py visualize --results-path outputs/small_strict_full/training_results.json --checkpoint-path outputs/small_strict_full/gppo_policy.pt