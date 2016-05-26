#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import glob
import shutil


def main():
    root_dir = os.getcwd()

    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if file == 'setup.py':
                cur_dir = os.path.abspath(root)
                print 'cd {}'.format(cur_dir)
                os.chdir(cur_dir)
                cmd = 'python {} build_ext --inplace'.format(file)
                print cmd
                p = subprocess.Popen(cmd)
                p.wait()

                lib_root = os.path.join(root,'ymh_mirlib')
                for root2,dirs2,files2 in os.walk(lib_root):
                    for _file in files2:
                        ext = os.path.splitext(_file)[1]
                        if ext == '.pyd':
                            print _file
                            print '{} -> {}'.format(root2, root)
                            shutil.copy2(os.path.join(root2,_file), root)

                if os.path.exists(lib_root):
                    shutil.rmtree(lib_root)


if __name__ == '__main__':
    main()
