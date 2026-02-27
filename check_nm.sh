#!/bin/bash
nm local/psblas3/lib/libpsb_cbind.a | grep psb_c_ > nm_output.txt
cat nm_output.txt
