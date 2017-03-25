#!/usr/bin/env bash
#
f_name='tss_ups500bp_dws500bp_win500bp_stride50bp'
python generate_regions.py ${f_name}
echo Sorting
bedtools sort -i ${f_name}.pos.bed> ${f_name}-sorted.pos.bed
bedtools sort -i ${f_name}.neg.bed > ${f_name}-sorted.neg.bed


echo performing positive strand: tssseq diamide
bwtool matrix 250:250 ${f_name}-sorted.pos.bed ../../data/bigwigs/tssseq/Dia.ts.pos.bw,../../data/bigwigs/tssseq/Dia.ts.neg.bw Dia.ts.sense_asense.txt
echo performing positive strand: tssseq diamide control
bwtool matrix 250:250 ${f_name}-sorted.pos.bed ../../data/bigwigs/tssseq/Dia_Cnt.ts.pos.bw,../../data/bigwigs/tssseq/Dia_Cnt.ts.neg.bw Dia_Cnt.ts.sense_asense.txt
echo performing positive strand: chipnexus diamide
bwtool matrix 250:250 ${f_name}-sorted.pos.bed ../../data/bigwigs/chipnexus/Dia.cn.pos.bw,../../data/bigwigs/chipnexus/Dia.cn.neg.bw Dia.cn.sense_asense.txt
echo performing positive strand: chipnexus diamide contol
bwtool matrix 250:250 ${f_name}-sorted.pos.bed ../../data/bigwigs/chipnexus/Dia_Cnt.cn.pos.bw,../../data/bigwigs/chipnexus/Dia_Cnt.cn.neg.bw Dia_Cnt.cn.sense_asense.txt

echo performing negative strand: tssseq diamide
bwtool matrix 250:250 ${f_name}-sorted.neg.bed ../../data/bigwigs/tssseq/Dia.ts.pos.bw,../../data/bigwigs/tssseq/Dia.ts.neg.bw Dia.ts.asense_sense_tbf.txt
echo performing negative strand: tssseq control
bwtool matrix 250:250 ${f_name}-sorted.neg.bed ../../data/bigwigs/tssseq/Dia_Cnt.ts.pos.bw,../../data/bigwigs/tssseq/Dia_Cnt.ts.neg.bw Dia_Cnt.ts.asense_sense_tbf.txt
echo performing negative strand: chipnexus diamide
bwtool matrix 250:250 ${f_name}-sorted.neg.bed ../../data/bigwigs/chipnexus/Dia.cn.pos.bw,../../data/bigwigs/chipnexus/Dia.cn.neg.bw Dia.cn.asense_sense_tbf.txt
echo performing negative strand: chipnexus diamide control
bwtool matrix 250:250 ${f_name}-sorted.neg.bed ../../data/bigwigs/chipnexus/Dia_Cnt.cn.pos.bw,../../data/bigwigs/chipnexus/Dia_Cnt.cn.neg.bw Dia_Cnt.cn.asense_sense_tbf.txt

echo performing fasta extraction
bedtools getfasta -fi ../../data/sacCer3.fa -bed ${f_name}-sorted.pos.bed -fo sense.fa
bedtools getfasta -fi ../../data/sacCer3.fa -bed ${f_name}-sorted.neg.bed -fo asense_tbf.fa

