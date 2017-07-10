#!/usr/bin/env bash

mysql --user=genome --host=genome-mysql.cse.ucsc.edu -A -D  -e sacCer3 \
'SELECT chrom,txStart,txStart,"TSS",".",strand FROM gene WHERE strand = "+";' \
| tail -n +2 > ../data/regions/tss.bed
