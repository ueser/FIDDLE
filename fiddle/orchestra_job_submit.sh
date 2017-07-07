to_submit='python main.py [options]'

bsub -q gpu -R "rusage[ngpus=2]" -W 11:59 -J experiment -u <your email>@email.com -N $to_submit -o %J.log -e %J.err
