#!/usr/bin/env python
import argparse
import sys

# torchlight
# import torchlight
from torchlight import import_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='llection')
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['ExplainerScrew_frame_sparsity_forgithub'] = import_class('processor.ExplainerScrew_frame_sparsity_forgithub.Explainer')#主要可运行实验

    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    arg = parser.parse_args()
    # Processor = processors[arg.processor]
    # Processor = processors['recognition']  
    Processor = processors['ExplainerScrew_frame_sparsity forgithub']  
    p = Processor(sys.argv[2:])
    p.start()
