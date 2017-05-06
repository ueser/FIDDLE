#!/usr/bin/env bash
#
f_name='tss_ups500bp_dws500bp_win500bp_stride50bp'
python generate_regions.py chr_size ${f_name} annotation
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


### for human enhancers



f_name='enhancer_strided'
python generate_regions.py chr_size ${f_name}.bed ../../data/FANTOM/regions/all_peaks_enhancer_custom_track.bed
cd ~/Projects/FIDDLE/data/regions
echo Sorting
bedtools sort -i ${f_name}.bed> ${f_name}-sorted.bed


echo performing on dnaseseq
bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/dnase_hela*.bigwig dnaseseq.tsv

echo performing on h3k4me1
bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/h3k4me1_hela*.bigwig h3k4me1.tsv

echo performing on h3k27ac
bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/h3k27ac_hela*.bigwig h3k27ac.tsv

echo performing on h3k27me3
bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/h3k27me3_hela*.bigwig h3k27me3.tsv

echo performing on netseq
bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/netseq_hela_signal.pos.bw,../../data/FANTOM/netseq_hela_signal.neg.bw netseq_pos_neg.tsv


echo performing fasta extraction
bedtools getfasta -fi ~/Projects/genome/data/annotation/hg19/hg19.fa -bed ${f_name}-sorted.bed -fo enhancers.fa


#### active vs inactive for testing

for f_name in 'active_enhancers' 'inactive_enhancers'
do
    python ~/Projects/FIDDLE/fiddle/data_prep/generate_regions.py chr_size ${f_name}.bed ${f_name}_.bed
    echo Sorting
    bedtools sort -i ${f_name}.bed> ${f_name}-sorted.bed
    echo performing on dnaseseq
    bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/dnase_hela*.bigwig dnaseseq_${f_name}.tsv

    echo performing on h3k4me1
    bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/h3k4me1_hela*.bigwig h3k4me1_${f_name}.tsv

    echo performing on h3k27ac
    bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/h3k27ac_hela*.bigwig h3k27ac_${f_name}.tsv

    echo performing on h3k27me3
    bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/h3k27me3_hela*.bigwig h3k27me3_${f_name}.tsv

    echo performing on netseq
    bwtool matrix 250:250 ${f_name}-sorted.bed ../../data/FANTOM/netseq_hela_signal.pos.bw,../../data/FANTOM/netseq_hela_signal.neg.bw netseq_pos_neg_${f_name}.tsv


    echo performing fasta extraction
    bedtools getfasta -fi ~/Projects/genome/data/annotation/hg19/hg19.fa -bed ${f_name}-sorted.bed -fo enhancers_${f_name}.fa

done