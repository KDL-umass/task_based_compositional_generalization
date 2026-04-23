#!/bin/bash

for f in  ./generated_jobs/sample_efficiency_2/*.sh; do sbatch "$f"; done
