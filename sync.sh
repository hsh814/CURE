#! /bin/bash
rsync -azvhP root@10.20.19.65:/root/alpha-repair/script/ ./alpha -e 'ssh -p 1602'