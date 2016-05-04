#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import os
import sys
import ConfigParser

import pydev

import numpy


class CircleData:
    def __init__(self, dim=2):
        self.__dim = dim

        # radius in [0.5, 0.7)
        self.__radius = numpy.random.random( (self.__dim, ) ) * 0.5 + 0.2
        print >> sys.stderr, 'Radius' + str(self.__radius)

    def generate_data(self, num, output_stream):
        for i in range(num):
            x = (numpy.random.random( (self.__dim, ) ) - 0.5) * 2.
            if (numpy.abs(x) <= self.__radius).all():
                y = 1.
            else:
                y = 0.
            print >> output_stream,  ','.join( map(lambda v:'%.3f'%v, x) ) +  ',%.0f' % y

if __name__=='__main__':
    config = ConfigParser.ConfigParser()
    config.read(sys.argv[1])
    
    # make global output path.
    output_path = config.get('output', 'path')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    

    for section in config.sections():
        if section == 'output':
            continue

        enable = int(config.get(section, 'enable'))
        if not enable:
            print >> sys.stderr, 'Ignore dataset [%s]' % section
            continue
        
        print >> sys.stderr, 'Process dataset [%s]' % section
        sub_path = config.get(section, 'path')
        target_path = output_path + '/' + sub_path

        # re-download whole dataset.
        os.system('rm -rf ' + target_path)
        os.mkdir(target_path)

        print >> sys.stderr, 'Change workdir to [%s]' % target_path
        os.chdir(target_path)
        for option in config.options(section):
            if option == 'path':
                continue

            value = config.get(section, option)
            if '###' in value:
                src, dest = value.split('###')
                print >> sys.stderr, 'Download data [%s] from [%s]' % (dest, src)
                os.system('wget "%s" -O "%s"' % (src, dest))
            else:
                print >> sys.stderr, 'Download data [%s]' % (value)
                os.system('wget "%s"' % (value))




