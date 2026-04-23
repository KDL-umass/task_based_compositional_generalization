#!/bin/bash

# for f in ./evaluation_jobs/disjoint2_3_diverse_fixed/*.sh; do sbatch "$f"; done
# for f in ./evaluation_jobs/disjoint2_3_diverse_fixed/*.sh; do sbatch "$f"; done

# for f in ./evaluation_jobs/coverage_6_uniform_fixed_all/*.sh; do sbatch "$f"; done
# for f in ./evaluation_jobs/reversecoverage_6_all_uniform_fixed/*.sh; do sbatch "$f"; done
# for f in ./evaluation_jobs/paircoverage_6_uniform_fixed_all/*.sh; do sbatch "$f"; done
for f in ./evaluation_jobs/disjoint7_6_diverse_nh6_nl3/*.sh; do sbatch "$f"; done
# ./evaluation_jobs/ndcontinuouscoverage_6_diverse_all/
# for f in ./generated_jobs/continuouspaircoverage_6_all_uniform_fixed_step_by_step/*.sh; do sbatch "$f"; done

# for f in ./evaluation_jobs/disjoint_3_diverse_fixed_all/*.sh; do sbatch "$f"; done
# for f in ./generated_jobs/disjoint2_6_diverse_fixed/*.sh; do sbatch "$f"; done