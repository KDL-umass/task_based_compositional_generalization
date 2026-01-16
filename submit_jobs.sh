#!/bin/bash

# for f in ./evaluation_jobs/disjoint2_3_diverse_fixed/*.sh; do sbatch "$f"; done
# for f in ./evaluation_jobs/disjoint2_3_diverse_fixed/*.sh; do sbatch "$f"; done

for f in ./evaluation_jobs/reversepaircoverage_6_0_uniform_fixed/*.sh; do sbatch "$f"; done
# for f in ./evaluation_jobs/disjoint_3_diverse_fixed_all/*.sh; do sbatch "$f"; done
# for f in ./generated_jobs/disjoint2_6_diverse_fixed/*.sh; do sbatch "$f"; done